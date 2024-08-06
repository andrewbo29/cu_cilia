import numpy as np
from pathlib import Path
from PIL import Image
from skimage.measure import label, regionprops, find_contours
from typing import List


def tiff_to_png(tiff_dir: str, png_dir: str):
    tiff_dir, png_dir = Path(tiff_dir), Path(png_dir)
    tiff_imgs = sorted(list(images_dir.rglob("*.tiff")) + list(images_dir.rglob("*.tif")))
    for tiff_path in tiff_imgs:
        relative_path = png_dir / tiff_path.relative_to(tiff_dir)
        png_path = relative_path.with_suffix(".png")
        png_path.parent.mkdir(exist_ok=True, parents=True)
        if png_path.exists():
            print(f"A png file already exists for {tiff_path.name}")
        else:
            im = Image.open(tiff_path)
            print(f"Generating png for {tiff_path.name}")
            im.save(png_path, "PNG", quality=100)

def yolobbox2bbox(yolo_bbox: List[float]) -> List[float]:
    x, y, w, h = yolo_bbox
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2

def mask_to_border(mask: np.ndarray) -> np.ndarray:
    border = np.zeros_like(mask, dtype=np.uint8)
    contours = find_contours(mask)
    for contour in contours:
        for c in contour:
            x, y = int(c[0]), int(c[1])
            border[x, y] = 255
    return border

def mask_to_bbox(mask: np.ndarray) -> List[List[int]]:
    bboxes = []
    border = mask_to_border(mask)
    labeled_img = label(border)
    props = regionprops(labeled_img)
    for prop in props:
        y1, x1, y2, x2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
    return bboxes
