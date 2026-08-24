# Copyright (c) 2026 Constructor Technology AG
#
# This file is part of cu_cilia, released under the GNU General Public
# License v3.0. See the LICENSE file for details.

import logging
import os
import pickle
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
from cellpose import core, models, plot, utils
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv("env.conf")
logger = logging.getLogger("nuclei segmentation")


def calculate_nuclei_features(
    imgs_to_resize: List[Path], masks: List[np.ndarray], output_dir: Path, img_size: Tuple[int, int] = (512, 512)
):
    """Calculates features of detected nuclei and saves the results.

    Args:
        imgs_to_resize: List of image paths.
        masks: List of segmentation masks.
        output_dir: Directory to save results.
        img_size: Target image size (height, width).
    """
    image_nuclei_data: Dict[str, List[Any]] = defaultdict(list)
    nuclei_data: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for idx in range(len(imgs_to_resize)):
        img_name = imgs_to_resize[idx].stem
        maski = masks[idx]
        cells_labels, counts = np.unique(np.int32(maski), return_counts=True)
        pred_cells = len(counts[1:])
        logger.debug(f"{imgs_to_resize[idx].stem}: {pred_cells}")

        sum_sq = 0
        for cell_count in counts[1:]:
            sum_sq += cell_count / (img_size[0] * img_size[1])
        if pred_cells == 0:
            mean_sq = 0.
        else:
            mean_sq = sum_sq / pred_cells

        image_nuclei_data["image_name"].append(img_name)
        image_nuclei_data["nuclei_number"].append(pred_cells)
        image_nuclei_data["sum_sq_perc"].append(sum_sq * 100)
        image_nuclei_data["mean_sq_perc"].append(mean_sq * 100)

        for cell in cells_labels[1:]:
            nuclei_data[img_name][cell] = {}
            cell_mask = maski == cell
            center_mass = scipy.ndimage.center_of_mass(cell_mask)
            cell_coords = []
            for i in range(img_size[0]):
                for j in range(img_size[1]):
                    if cell_mask[i, j]:
                        cell_coords.append((i, j))

            nuclei_data[img_name][cell]["center_mass"] = center_mass
            nuclei_data[img_name][cell]["coords"] = cell_coords

    df = pd.DataFrame(image_nuclei_data)
    df.to_csv(output_dir / "image_nuclei_features.csv", index=False)

    with open(output_dir / "nuclei_features.pkl", "wb") as handle:
        pickle.dump(nuclei_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_outlines(img: np.ndarray, mask: np.ndarray, channels: List[int] = [3, 0]) -> np.ndarray:
    """Generates an outlined image with cell masks highlighted in red.

    Args:
        img: Input image.
        mask: Segmentation mask.
        channels: Image channels for visualization.

    Returns:
        Image with outlined segmentation masks.
    """
    img0 = img.copy()

    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = plot.image_to_rgb(img0, channels=channels)
    else:
        if img0.max() <= 50.0:
            img0 = np.clip(img0 * 255, 0, 1).astype(np.uint8)

    outlines = utils.masks_to_outlines(mask)
    outX, outY = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    return imgout


def main():
    """Main function to execute nuclei segmentation using Cellpose, calculate statistics, and save the results."""
    images_dir = Path(os.environ.get("IMAGES_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR")) / "nuclei_results"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    img_size = [512, 512]
    channels = [3, 0]

    logger.info("Running nuclei detection model")
    imgs_to_resize = sorted(
        list(images_dir.rglob("*.tiff")) + list(images_dir.rglob("*.tif"))
    )
    resized_images = []
    for img_path in imgs_to_resize:
        img = skimage.io.imread(img_path)
        if img.size == 0 or len(img.shape) < 2:
            logger.debug(
                f"Skipping {img_path} due to empty image or wrong format. Using black image instead."
            )
            img = np.zeros((img_size[0], img_size[1], 3))
        img_resize = skimage.transform.resize(img, img_size)
        img_resize = skimage.util.img_as_ubyte(img_resize)
        resized_images.append(img_resize)

    use_GPU = core.use_gpu()
    logger.debug(f">>> GPU activated? {bool(use_GPU)}")

    model_path = "models/cellpose_thyroid/models/cellpose_thyroid"
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks = model.eval(
        resized_images,
        channels=channels,
        diameter=model.diam_labels,
        flow_threshold=float(os.environ.get("CELLPOSE_FLOW_THRESHOLD", 0.4)),
        cellprob_threshold=float(os.environ.get("CELLPOSE_CELLPROB_THRESHOLD", 0.0)),
        min_size=float(os.environ.get("CELLPOSE_MIN_SIZE", 15.0)),
    )[0]
    if os.environ.get("EXCLUDE_EDGE_NUCLEI", "False") == "True":
        masks = [utils.remove_edge_masks(mask) for mask in masks]

    logger.info("Calculating nuclei statistics")
    calculate_nuclei_features(imgs_to_resize, masks, output_dir)

    logger.info("Saving masks")
    seg_images_folder = output_dir / "nuclei"
    if seg_images_folder.exists():
        shutil.rmtree(seg_images_folder)
    seg_images_folder.mkdir(exist_ok=True, parents=True)

    for k in range(len(resized_images)):
        im = resized_images[k]
        img = im.copy()
        cell_image = make_outlines(img, masks[k], channels)
        output_name = (seg_images_folder / imgs_to_resize[k].stem).with_suffix(".png")
        plt.imsave(output_name, cell_image)


if __name__ == "__main__":
    main()
