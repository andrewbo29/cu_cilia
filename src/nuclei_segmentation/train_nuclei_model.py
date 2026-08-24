# Copyright (c) 2026 Constructor Technology AG
#
# This file is part of cu_cilia, released under the GNU General Public
# License v3.0. See the LICENSE file for details.

import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import yaml
from cellpose import io, metrics, models
from dotenv import load_dotenv

load_dotenv(".env")  # loading mlflow credentials
logger = logging.getLogger(__name__)


def read_exp_config():
    code_path = Path(__file__).parent
    config_path = code_path / "configs/train_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(config)

    return config


CONFIG = read_exp_config()


def mean_cells_ratio(gt, pred):
    if len(gt) != len(pred):
        ValueError(f"Length gt {len(gt)} and pred {len(pred)} must be equal")
    ratios = []
    for i in range(len(gt)):
        _, counts = np.unique(np.int32(pred[i]), return_counts=True)
        pred_cells = len(counts[1:])
        _, counts = np.unique(np.int32(gt[i]), return_counts=True)
        gt_cells = len(counts[1:])
        ratios.append(pred_cells / gt_cells)

    return np.mean(ratios)


def train_nuclei_model():
    root_path = Path(__file__).parent.parent.parent.resolve()

    out_dir = root_path / "models"
    os.makedirs(out_dir, exist_ok=True)

    data_path = root_path / "data/nuclei_dataset"

    mlflow.set_experiment("va_nuclei_segmentation")

    if CONFIG["not_treated_exp"] and CONFIG["n_support"] == -1:
        version = "not_treated"
    elif not CONFIG["not_treated_exp"] and CONFIG["n_support"] == -1:
        version = "treated"
    elif CONFIG["not_treated_exp"] and CONFIG["n_support"] != -1:
        version = f"not_treated_fsl_{CONFIG['n_support']}"
    elif not CONFIG["not_treated_exp"] and CONFIG["n_support"] != -1:
        version = f"treated_fsl_{CONFIG['n_support']}"

    template_name = f"va_nuclei_segmentation_{version}"

    train_dirs_list = []
    if CONFIG["not_treated_exp"]:
        train_dirs_list.append(
            os.path.join(data_path, "train_by_exps", "Nthyori,exp3,cond3")
        )
        train_dirs_list.append(
            os.path.join(data_path, "train_by_exps", "Nthyori,exp5,cond1a,control")
        )
    else:
        train_dirs_list.append(
            os.path.join(data_path, "train_by_exps", "Nthyori,exp3,cond3")
        )
        train_dirs_list.append(
            os.path.join(data_path, "train_by_exps", "Nthyori,exp5,cond1a,control")
        )
        train_dirs_list.append(
            os.path.join(data_path, "train_by_exps", "Nthyori,exp4,cond3")
        )
    test_dir = os.path.join(data_path, "test")

    test_experiment_types = defaultdict(list)
    p = Path(test_dir)
    for i, fname in enumerate(p.rglob("*_seg.npy")):
        exp_name = ",".join(fname.stem.split(",")[1:3])
        test_experiment_types[exp_name].append(i)

    model_type = "cyto2"
    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    n_epochs = CONFIG["n_epochs"]
    nimg_per_epoch = CONFIG["nimg_per_epoch"]
    learning_rate = CONFIG["learning_rate"]
    weight_decay = CONFIG["weight_decay"]
    batch_size = CONFIG["batch_size"]

    model_name = "cellpose_thyroid"
    model_path = os.path.join(out_dir, model_name)

    train_data_list = []
    train_labels_list = []
    for train_dir in train_dirs_list:
        logger.debug(train_dir)
        output = io.load_train_test_data(train_dir, None, mask_filter="_seg.npy")
        train_data, train_labels, _, _, _, _ = output
        train_data_list.append(np.array(train_data))
        train_labels_list.append(np.array(train_labels))

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_labels_list)

    model = models.CellposeModel(gpu=True, model_type=model_type)

    logger.info("Train")
    new_model_path = model.train(
        list(train_data),
        list(train_labels),
        test_data=None,
        test_labels=None,
        channels=channels,
        save_path=model_path,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        nimg_per_epoch=nimg_per_epoch,
        batch_size=batch_size,
        model_name=model_name,
    )

    model = models.CellposeModel(gpu=True, pretrained_model=new_model_path)
    diameter = 0
    diameter = model.diam_labels if diameter == 0 else diameter

    logger.info("Test")
    test_dir = os.path.join(data_path, "test")
    output = io.load_train_test_data(test_dir, mask_filter="_seg.npy")
    test_data, test_labels = output[:2]

    masks = model.eval(test_data, channels=[chan, chan2], diameter=diameter)[0]

    logger.info("Calculate metrics")
    ap = metrics.average_precision(test_labels, masks)[0]
    ap_05 = ap[:, 0].mean()
    ap_075 = ap[:, 1].mean()
    cnr_val = mean_cells_ratio(gt=test_labels, pred=masks)
    cnr = abs(1 - cnr_val)

    epoch_loss = parse_logger()

    with mlflow.start_run(run_name=template_name) as run:
        mlflow.log_param("experiment", template_name)

        mlflow.log_param("n_epochs", CONFIG["n_epochs"])
        mlflow.log_param("nimg_per_epoch", CONFIG["nimg_per_epoch"])
        mlflow.log_param("learning_rate", CONFIG["learning_rate"])
        mlflow.log_param("weight_decay", CONFIG["weight_decay"])
        mlflow.log_param("batch_size", CONFIG["batch_size"])

        mlflow.log_metric("Test AP IoU 0.5", ap_05)
        mlflow.log_metric("Test AP IoU 0.75", ap_075)
        mlflow.log_metric("Test mean abs 1 minus CNR", cnr)

        for ep, loss_val in epoch_loss.items():
            mlflow.log_metric("Train loss", loss_val, step=ep)

        logger.info("Collecting artifacts")
        artifacts_folder = os.path.join(out_dir, "artifacts")
        os.makedirs(artifacts_folder, exist_ok=True)
        artifacts_model_path = os.path.join(artifacts_folder, "model")
        os.makedirs(artifacts_model_path, exist_ok=True)
        shutil.copy(
            new_model_path, os.path.join(artifacts_model_path, "cellpose_thyroid")
        )

        code_path = Path(__file__).parent
        config_path = code_path / "configs/train_config.yaml"
        shutil.copy(
            config_path, os.path.join(artifacts_folder, "configs/train_config.yaml")
        )

        mlflow.log_artifacts(artifacts_folder)

        shutil.rmtree(artifacts_folder)


def parse_logger():
    logger_fname = "/home/coder/.cellpose/run.log"

    epoch_loss = {}
    with open(logger_fname, "r") as f:
        for line in f:
            split_res = line.rstrip("\n").split(" ")
            if "Loss" in split_res:
                for i in range(len(split_res)):
                    if split_res[i] == "Epoch":
                        epoch_val = int(split_res[i + 1][:-1])
                    if split_res[i] == "Loss":
                        loss_val = float(split_res[i + 1][:-1])
                        break
                epoch_loss[epoch_val] = loss_val

    return epoch_loss


def main():
    train_nuclei_model()


if __name__ == "__main__":
    sys.exit(main())
