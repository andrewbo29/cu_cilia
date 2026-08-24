# Copyright (c) 2026 Constructor Technology AG
#
# This file is part of cu_cilia, released under the GNU General Public
# License v3.0. See the LICENSE file for details.

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from cellpose import io, metrics, models
from tqdm import tqdm

logger = logging.getLogger(__name__)


FSL_NOT_TREATED_CONFIG = {
    1: {
        "n_support": 1,
        "n_epochs": 50,
        "nimg_per_epoch": 35,
        "learning_rate": 0.001352,
        "weight_decay": 0.0001265,
        "batch_size": 4,
    },
    2: {
        "n_support": 2,
        "n_epochs": 50,
        "nimg_per_epoch": 35,
        "learning_rate": 0.004179,
        "weight_decay": 0.0002381,
        "batch_size": 4,
    },
    3: {
        "n_support": 3,
        "n_epochs": 20,
        "nimg_per_epoch": 35,
        "learning_rate": 0.04153,
        "weight_decay": 0.00001082,
        "batch_size": 1,
    },
    4: {
        "n_epochs": 30,
        "nimg_per_epoch": 20,
        "learning_rate": 0.003636,
        "weight_decay": 0.02761,
        "batch_size": 1,
        "n_support": 4,
    },
    5: {
        "n_epochs": 30,
        "nimg_per_epoch": 35,
        "learning_rate": 0.0248,
        "weight_decay": 1e-5,
        "batch_size": 8,
        "n_support": 5,
    },
}

FSL_TREATED_CONFIG = {
    1: {
        "n_epochs": 20,
        "nimg_per_epoch": 35,
        "learning_rate": 0.3968,
        "weight_decay": 0.00001389,
        "batch_size": 4,
        "n_support": 1,
    },
    2: {
        "n_epochs": 20,
        "nimg_per_epoch": 30,
        "learning_rate": 0.0334,
        "weight_decay": 0.0001636,
        "batch_size": 1,
        "n_support": 2,
    },
    3: {
        "n_epochs": 70,
        "nimg_per_epoch": 5,
        "learning_rate": 0.01016,
        "weight_decay": 0.008228,
        "batch_size": 8,
        "n_support": 3,
    },
    4: {
        "n_epochs": 70,
        "nimg_per_epoch": 5,
        "learning_rate": 0.01757,
        "weight_decay": 0.001002,
        "batch_size": 8,
        "n_support": 4,
    },
    5: {
        "n_epochs": 70,
        "nimg_per_epoch": 5,
        "learning_rate": 0.01237,
        "weight_decay": 0.0005402,
        "batch_size": 8,
        "n_support": 5,
    },
}


def read_exp_config():
    code_path = Path(__file__).parent
    config_path = code_path / "configs/test_config.yaml"

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


def test_model(is_workflow=False):
    root_path = Path(__file__).parent.parent.parent.resolve()

    if is_workflow:
        out_dir = "/output/atmc_test_results"
    else:
        out_dir = root_path / "data" / "atmc_test_results"
        os.makedirs(out_dir, exist_ok=True)

    if CONFIG["not_treated_exp"]:
        train_dir = root_path / "data" / "nuclei_dataset" / "train_set_not_treated"
    else:
        train_dir = root_path / "data" / "nuclei_dataset" / "train_set"
    test_dir = root_path / "data" / "nuclei_dataset" / "test"

    test_experiment_types = defaultdict(list)
    p = Path(test_dir)
    for i, fname in enumerate(p.rglob("*_seg.npy")):
        exp_name = ",".join(fname.stem.split(",")[1:3])
        test_experiment_types[exp_name].append(i)

    model_type = "cyto2"

    n_epochs = CONFIG["n_epochs"]
    nimg_per_epoch = CONFIG["nimg_per_epoch"]
    learning_rate = CONFIG["learning_rate"]
    weight_decay = CONFIG["weight_decay"]
    batch_size = CONFIG["batch_size"]

    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    model_name = "adapted_thyroid_mask_constructor"
    model_path = os.path.join(out_dir, model_name)

    ap_05 = defaultdict(list)
    ap_075 = defaultdict(list)
    cnr = defaultdict(list)

    logger.debug(str(train_dir))
    logger.debug(str(test_dir))

    for _ in tqdm(range(CONFIG["tries_num"]), desc="Try"):
        output = io.load_train_test_data(
            str(train_dir), str(test_dir), mask_filter="_seg.npy"
        )
        train_data, train_labels, _, test_data, test_labels, _ = output

        model = models.CellposeModel(gpu=True, model_type=model_type)

        logger.info("Train")
        new_model_path = model.train(
            train_data,
            train_labels,
            test_data=test_data,
            test_labels=test_labels,
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
        output = io.load_train_test_data(str(test_dir), mask_filter="_seg.npy")
        test_data, test_labels = output[:2]

        masks = model.eval(test_data, channels=[chan, chan2], diameter=diameter)[0]

        logger.info("Calculate metrics")
        ap = metrics.average_precision(test_labels, masks)[0]
        ap_05["all"].append(ap[:, 0].mean())
        ap_075["all"].append(ap[:, 1].mean())
        cnr_val = mean_cells_ratio(gt=test_labels, pred=masks)
        cnr["all"].append(abs(1 - cnr_val))

        for exp_name in test_experiment_types:
            exp_img_inds = np.array(test_experiment_types[exp_name])
            ap_05[exp_name].append(ap[exp_img_inds, 0].mean())
            ap_075[exp_name].append(ap[exp_img_inds, 1].mean())
            cnr_val = mean_cells_ratio(
                gt=np.array(test_labels)[exp_img_inds],
                pred=np.array(masks)[exp_img_inds],
            )
            cnr[exp_name].append(abs(1 - cnr_val))

    np.save(os.path.join(str(out_dir), "ap_05_all.npy"), np.array(ap_05["all"]))
    np.save(os.path.join(str(out_dir), "ap_075_all.npy"), np.array(ap_075["all"]))
    np.save(os.path.join(str(out_dir), "cnr_all.npy"), np.array(cnr["all"]))

    for exp_name in test_experiment_types:
        np.save(
            os.path.join(str(out_dir), f"ap_05_{exp_name}.npy"),
            np.array(ap_05[exp_name]),
        )
        np.save(
            os.path.join(str(out_dir), f"ap_075_{exp_name}.npy"),
            np.array(ap_075[exp_name]),
        )
        np.save(
            os.path.join(str(out_dir), f"cnr_{exp_name}.npy"), np.array(cnr[exp_name])
        )


def test_fsl(is_workflow=False):
    if CONFIG["not_treated_exp"]:
        config = FSL_NOT_TREATED_CONFIG
    else:
        config = FSL_TREATED_CONFIG

    root_path = Path(__file__).parent.parent.parent.resolve()

    if is_workflow:
        out_dir = "/output/atmc_fsl_test_results"
    else:
        out_dir = root_path / "data" / "atmc_fsl_test_results"
        os.makedirs(out_dir, exist_ok=True)

    not_treated_data = False

    data_path = root_path / "data" / "nuclei_dataset" / "train_by_exps"
    data_path = str(data_path)

    train_dirs_list = []
    if CONFIG["not_treated_exp"]:
        train_dirs_list.append(os.path.join(data_path, "Nthyori,exp3,cond3"))
        train_dirs_list.append(os.path.join(data_path, "Nthyori,exp5,cond1a,control"))
    else:
        train_dirs_list.append(os.path.join(data_path, "Nthyori,exp3,cond3"))
        train_dirs_list.append(os.path.join(data_path, "Nthyori,exp5,cond1a,control"))
        train_dirs_list.append(os.path.join(data_path, "Nthyori,exp4,cond3"))
    test_dir = root_path / "data" / "nuclei_dataset" / "test"

    test_experiment_types = defaultdict(list)
    p = Path(test_dir)
    for i, fname in enumerate(p.rglob("*_seg.npy")):
        exp_name = ",".join(fname.stem.split(",")[1:3])
        test_experiment_types[exp_name].append(i)

    model_type = "cyto2"

    n_support = config[CONFIG["n_support"]]["n_support"]
    n_epochs = config[CONFIG["n_support"]]["n_epochs"]
    nimg_per_epoch = config[CONFIG["n_support"]]["nimg_per_epoch"]
    learning_rate = config[CONFIG["n_support"]]["learning_rate"]
    weight_decay = config[CONFIG["n_support"]]["weight_decay"]
    batch_size = config[CONFIG["n_support"]]["batch_size"]

    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    model_name = "adapted_thyroid_mask_constructor"
    model_path = os.path.join(out_dir, model_name)

    train_data_list = []
    train_labels_list = []

    for train_dir in train_dirs_list:
        output = io.load_train_test_data(train_dir, None, mask_filter="_seg.npy")
        train_data, train_labels, _, _, _, _ = output
        train_data_list.append(train_data)
        train_labels_list.append(train_labels)

    output = io.load_train_test_data(str(test_dir), mask_filter="_seg.npy")
    test_data, test_labels = output[:2]

    train_inds_list = []
    for i in range(len(train_data_list)):
        train_inds = np.arange(len(train_data_list[i]))
        train_inds_list.append(train_inds)

    test_metrics = {}
    test_metrics["all"] = defaultdict(list)
    for exp_name in test_experiment_types:
        test_metrics[exp_name] = defaultdict(list)

    for __ in tqdm(range(CONFIG["tries_num"]), desc="Few-shot try"):
        train_data_fold, train_labels_fold = [], []
        for j in range(len(train_inds_list)):
            support_ind = np.random.choice(train_inds_list[j], n_support, replace=False)
            train_data_fold.append(np.array(train_data_list[j])[support_ind])
            train_labels_fold.append(np.array(train_labels_list[j])[support_ind])

        train_data_fold = np.concatenate(train_data_fold)
        train_labels_fold = np.concatenate(train_labels_fold)

        model = models.CellposeModel(gpu=True, model_type=model_type)

        trained_model_path = model.train(
            list(train_data_fold),
            list(train_labels_fold),
            test_data=test_data,
            test_labels=test_labels,
            channels=channels,
            save_path=model_path,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            nimg_per_epoch=nimg_per_epoch,
            model_name=model_name,
        )

        model = models.CellposeModel(gpu=True, pretrained_model=trained_model_path)
        diameter = 0
        diameter = model.diam_labels if diameter == 0 else diameter

        output = io.load_train_test_data(str(test_dir), mask_filter="_seg.npy")
        test_data, test_labels = output[:2]

        masks = model.eval(test_data, channels=[chan, chan2], diameter=diameter)[0]

        ap = metrics.average_precision(test_labels, masks)[0]
        test_metrics["all"]["ap_0.5"].append(ap[:, 0].mean())
        test_metrics["all"]["ap_0.75"].append(ap[:, 1].mean())
        cnr_val = mean_cells_ratio(gt=test_labels, pred=masks)
        test_metrics["all"]["cell_ratio"].append(abs(1 - cnr_val))

        for exp_name in test_experiment_types:
            exp_img_inds = np.array(test_experiment_types[exp_name])
            test_metrics[exp_name]["ap_0.5"].append(ap[exp_img_inds, 0].mean())
            test_metrics[exp_name]["ap_0.75"].append(ap[exp_img_inds, 1].mean())
            cnr_val = mean_cells_ratio(
                gt=np.array(test_labels)[exp_img_inds],
                pred=np.array(masks)[exp_img_inds],
            )
            test_metrics[exp_name]["cell_ratio"].append(abs(1 - cnr_val))

    np.save(
        os.path.join(str(out_dir), "ap_05_all.npy"),
        np.array(test_metrics["all"]["ap_0.5"]),
    )
    np.save(
        os.path.join(str(out_dir), "ap_075_all.npy"),
        np.array(test_metrics["all"]["ap_0.75"]),
    )
    np.save(
        os.path.join(str(out_dir), "cnr_all.npy"),
        np.array(test_metrics["all"]["cell_ratio"]),
    )

    for exp_name in test_experiment_types:
        np.save(
            os.path.join(str(out_dir), f"ap_05_{exp_name}.npy"),
            np.array(test_metrics[exp_name]["ap_0.5"]),
        )
        np.save(
            os.path.join(str(out_dir), f"ap_075_{exp_name}.npy"),
            np.array(test_metrics[exp_name]["ap_0.75"]),
        )
        np.save(
            os.path.join(str(out_dir), f"cnr_{exp_name}.npy"),
            np.array(test_metrics[exp_name]["cell_ratio"]),
        )


def test_old_model(is_workflow=False):
    root_path = Path(__file__).parent.parent.parent.resolve()

    if is_workflow:
        out_dir = "/output/old_test_results"
    else:
        out_dir = root_path / "data" / "old_test_results"
        os.makedirs(out_dir, exist_ok=True)

    test_dir = root_path / "data" / "nuclei_dataset" / "test"

    test_experiment_types = defaultdict(list)
    p = Path(test_dir)
    for i, fname in enumerate(p.rglob("*_seg.npy")):
        exp_name = ",".join(fname.stem.split(",")[1:3])
        test_experiment_types[exp_name].append(i)

    test_dir = str(test_dir)

    model_type = "cyto2"

    chan = 3
    chan2 = 0
    channels = [chan, chan2]

    ap_05 = defaultdict(list)
    ap_075 = defaultdict(list)
    cnr = defaultdict(list)

    for _ in tqdm(range(CONFIG["tries_num"]), desc="Try"):
        model = models.Cellpose(gpu=True, model_type=model_type, net_avg=True)

        output = io.load_train_test_data(test_dir, mask_filter="_seg.npy")
        test_data, test_labels = output[:2]

        diameter = 0
        # diameter = model.diam_labels if diameter==0 else diameter

        masks = model.eval(test_data, channels=[chan, chan2], diameter=diameter)[0]

        ap = metrics.average_precision(test_labels, masks)[0]
        ap_05["all"].append(ap[:, 0].mean())
        ap_075["all"].append(ap[:, 1].mean())
        cnr_val = mean_cells_ratio(gt=test_labels, pred=masks)
        cnr["all"].append(abs(1 - cnr_val))

        for exp_name in test_experiment_types:
            exp_img_inds = np.array(test_experiment_types[exp_name])
            ap_05[exp_name].append(ap[exp_img_inds, 0].mean())
            ap_075[exp_name].append(ap[exp_img_inds, 1].mean())
            cnr_val = mean_cells_ratio(
                gt=np.array(test_labels)[exp_img_inds],
                pred=np.array(masks)[exp_img_inds],
            )
            cnr[exp_name].append(abs(1 - cnr_val))

    np.save(os.path.join(str(out_dir), "ap_05_all.npy"), np.array(ap_05["all"]))
    np.save(os.path.join(str(out_dir), "ap_075_all.npy"), np.array(ap_075["all"]))
    np.save(os.path.join(str(out_dir), "cnr_all.npy"), np.array(cnr["all"]))

    for exp_name in test_experiment_types:
        np.save(
            os.path.join(str(out_dir), f"ap_05_{exp_name}.npy"),
            np.array(ap_05[exp_name]),
        )
        np.save(
            os.path.join(str(out_dir), f"ap_075_{exp_name}.npy"),
            np.array(ap_075[exp_name]),
        )
        np.save(
            os.path.join(str(out_dir), f"cnr_{exp_name}.npy"), np.array(cnr[exp_name])
        )


def main():
    # test_model(is_workflow=True)
    test_fsl(is_workflow=True)
    # test_old_model(is_workflow=False)


if __name__ == "__main__":
    sys.exit(main())
