# Copyright (c) 2026 Constructor Technology AG
#
# This file is part of cu_cilia, released under the GNU General Public
# License v3.0. See the LICENSE file for details.

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.measure import find_contours, label, regionprops

logger = logging.getLogger(__name__)


def tiff_to_png(tiff_dir: Path, png_dir: Path):
    """Converts TIFF images to PNG format and saves them in the specified directory.

    Args:
        tiff_dir: Directory containing TIFF images.
        png_dir: Target directory to save PNG images.
    """
    tiff_imgs = sorted(list(tiff_dir.rglob("*.tiff")) + list(tiff_dir.rglob("*.tif")))
    for tiff_path in tiff_imgs:
        relative_path = png_dir / tiff_path.relative_to(tiff_dir)
        png_path = relative_path.with_suffix(".png")
        png_path.parent.mkdir(exist_ok=True, parents=True)
        if png_path.exists():
            logger.info(f"A png file already exists for {tiff_path.name}")
        else:
            im = Image.open(tiff_path)
            logger.info(f"Generating png for {tiff_path.name}")
            im.save(png_path, "PNG", quality=100)


def yolobbox2bbox(yolo_bbox: List[float]) -> Tuple[float, float, float, float]:
    """Converts YOLO format bounding box to standard bounding box coordinates.

    Args:
        yolo_bbox: List containing YOLO format bounding box [x, y, w, h].

    Returns:
        Tuple representing standard bounding box (x1, y1, x2, y2).
    """
    x, y, w, h = yolo_bbox
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


def mask_to_bbox(mask: np.ndarray) -> List[List[int]]:
    """Converts a binary mask to bounding boxes.

    Args:
        mask: Binary mask as a NumPy array.

    Returns:
        List of bounding boxes in [x1, y1, x2, y2] format.
    """
    bboxes = []
    border = _mask_to_border(mask)
    labeled_img = label(border)
    props = regionprops(labeled_img)
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
    return bboxes


def _mask_to_border(mask: np.ndarray) -> np.ndarray:
    """Extracts the border of a binary mask.

    Args:
        mask: Binary mask as a NumPy array.

    Returns:
        Border mask as a NumPy array.
    """
    border = np.zeros_like(mask, dtype=np.uint8)
    contours = find_contours(mask)
    for contour in contours:
        for c in contour:
            x, y = int(c[0]), int(c[1])
            border[x, y] = 255
    return border
