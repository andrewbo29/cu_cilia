# Copyright (c) 2026 Constructor Technology AG
#
# This file is part of cu_cilia, released under the GNU General Public
# License v3.0. See the LICENSE file for details.

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
import skimage
from dotenv import load_dotenv
from fil_finder import FilFinder2D

load_dotenv("env.conf")
logger = logging.getLogger(__name__)


@dataclass
class ImageLoader:
    """Loads image from path."""
    def load_image(self, img_path: Path) -> np.ndarray:
        """Loads an image from a specified path.

        Args:
            img_path: Path to the image file.

        Returns:
            Loaded image as a NumPy array, or a black image if loading fails.
        """
        img = cv2.imread(img_path.as_posix())
        if img is None:
            logger.debug(
                f"Image {img_path} is not found. Using black image instead.",
            )
            return np.zeros((512, 512, 3))
        return img


@dataclass
class ImageProcessor:
    """Processes images using various filtering and thresholding techniques."""

    gauss_sigma: float = float(os.environ.get("CILIA_GAUSS_SIGMA", 8))
    disk_radius: int = int(os.environ.get("CILIA_TOPHAT_RADIUS", 13))
    debug: bool = os.environ.get("DEBUG", "False") == "True"

    def preprocess_image(
        self, rg_image: np.ndarray, output_dir: Path, img_path: Path
    ) -> np.ndarray:
        """Preprocesses an image using CLAHE, Gaussian blur, top-hat filtering, thresholding, and morphological operations.

        Args:
            rg_image: Input image.
            output_dir: Directory to save interim results.
            img_path: Path of the input image.

        Returns:
            Preprocessed binary image.
        """
        def save_interim_results(img: np.ndarray, dir_name: str):
            if self.debug:
                filename = output_dir / dir_name / f"{img_path.stem}.png"
                filename.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename.as_posix(), img)

        save_interim_results(rg_image, "0_rg")

        clahe = skimage.exposure.equalize_adapthist(rg_image)
        save_interim_results((clahe * 255).astype(np.uint8), "1_clahe")

        blurred = skimage.filters.gaussian(clahe, sigma=self.gauss_sigma)
        save_interim_results((blurred * 255).astype(np.uint8), "2_blurred")

        footprint = skimage.morphology.disk(self.disk_radius)
        tophat = skimage.morphology.white_tophat(blurred, footprint)
        save_interim_results((tophat * 255).astype(np.uint8), "3_tophat")

        thresh = skimage.filters.threshold_yen(tophat)
        thresholded = (tophat > thresh).astype(np.uint8) * 255
        save_interim_results(thresholded, "4_thresholded")

        kernel = np.ones((5, 5), np.uint8)
        preprocessed_img = cv2.morphologyEx(
            thresholded, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        save_interim_results(preprocessed_img, "5_preprocessed_img")

        return preprocessed_img


@dataclass
class ImageAnalyzer:
    """Analyzes images to extract features such as area, perimeter, and eccentricity."""

    min_length: float = float(os.environ.get("CILIA_MIN_LENGTH", 14))
    min_area: float = float(os.environ.get("CILIA_MIN_AREA", 256))
    min_eccentricity: float = float(os.environ.get("CILIA_MIN_ECCENTRICITY", 0.46))
    min_perimeter: float = float(os.environ.get("CILIA_MIN_PERIMETER", 67))
    exclude_edge_cilia: bool = os.environ.get("EXCLUDE_EDGE_CILIA", "False") == "True"
    edge_margin: int = 3

    def get_props(
        self, img_path: Path, preprocessed_img: np.ndarray, original_img: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extracts image properties and calculates additional features.

        Args:
            img_path: Path of the image file.
            preprocessed_img: Binary processed image.
            original_img: Original input image.

        Returns:
            Tuple containing labeled image and DataFrame of extracted properties.
        """
        label_img = skimage.measure.label(preprocessed_img)
        props = skimage.measure.regionprops_table(
            label_img,
            preprocessed_img,
            properties=(
                "label",
                "area",
                "orientation",
                "axis_major_length",
                "axis_minor_length",
                "bbox",
                "eccentricity",
                "image",
                "perimeter",
            ),
        )
        data = pd.DataFrame(props)
        data.columns = [col.replace("-", "_") for col in data.columns]
        data = self.filter(data, original_img.shape)
        label_img, data = self.postprocess(data, label_img)
        if len(data) > 1:
            data["skeleton_length"] = self.calculate_skeleton_length(data)
            data["area"] = self.calculate_true_area(data, original_img)
            data["image_name"] = img_path.stem

        return label_img, data.round(2)

    def filter(self, data: pd.DataFrame, img_shape: Tuple[int, ...]) -> pd.DataFrame:
        """
        Filters out objects that do not meet the specified criteria.

        Args:
            data: DataFrame containing image properties.
            img_shape: Shape of the original image.

        Returns:
            Filtered DataFrame.
        """
        if self.exclude_edge_cilia:
            data = data[
                (data.bbox_0 > self.edge_margin)
                & (data.bbox_1 > self.edge_margin)
                & (data.bbox_2 < img_shape[0] - self.edge_margin)
                & (data.bbox_3 < img_shape[1] - self.edge_margin)
            ]
        return data[
            (data.axis_major_length >= self.min_length)
            & (data.eccentricity >= self.min_eccentricity)
            & (data.area >= self.min_area)
            & (data.perimeter >= self.min_perimeter)
        ]

    def postprocess(
        self, props: pd.DataFrame, label_img: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Post-processes labeled image properties.

        Args:
            props: DataFrame containing extracted image properties.
            label_img: Labeled image.

        Returns:
            Updated labeled image and DataFrame with processed properties.
        """
        mapping = {j: i for i, j in enumerate(props.label.tolist(), start=1)}
        props["old_label"] = props.label
        props["label"] = props.old_label.apply(mapping.get)
        props["form_factor"] = 4 * np.pi * props.area / props.perimeter**2

        float_columns = props.select_dtypes(include=[float]).columns
        props[float_columns] = props[float_columns].round(2)

        label_img = np.vectorize(mapping.get, otypes=[int])(label_img, 0)

        return label_img, props

    def calculate_skeleton_length(self, props: pd.DataFrame) -> List[float]:
        """Calculates the skeleton length for each detected object.

        Args:
            props: DataFrame containing extracted image properties.

        Returns:
            List of skeleton lengths.
        """
        lengths = []
        for row in props.itertuples():
            im = row.image.astype(np.uint8)
            try:
                fil = FilFinder2D(im, mask=im)
                fil.preprocess_image(skip_flatten=True)
                fil.create_mask(
                    border_masking=True, verbose=False, use_existing_mask=True
                )
                fil.medskel(verbose=False)
                fil.analyze_skeletons(
                    skel_thresh=self.min_length * u.pix,
                    prune_criteria="length",
                    verbose=False,
                )
            except (TypeError, ValueError):
                fil.number_of_filaments = 0

            if fil.number_of_filaments == 0:
                skeleton_length = row.axis_major_length
                logger.debug(f"No filaments found for cilia {row.label}")
            else:
                end_coords = [
                    coord
                    for coord in fil.end_pts[0]
                    if fil.skeleton_longpath[coord] > 0
                ]
                if len(end_coords) > 1:
                    length_addition = (
                        fil.medial_axis_distance[end_coords[0]].value
                        + fil.medial_axis_distance[end_coords[1]].value
                    )
                    skeleton_length = sorted(fil.lengths().value)[-1] + length_addition
                else:
                    skeleton_length = row.axis_major_length
            lengths.append(skeleton_length)
        return lengths

    def calculate_true_area(
        self, data: pd.DataFrame, original_img: np.ndarray
    ) -> List[int]:
        """Calculates the true area of detected objects.

        Args:
            data: DataFrame containing extracted image properties.
            original_img: Original image.

        Returns:
            List of calculated areas.
        """
        areas = []
        indent = 2
        for row in data.itertuples():
            min_y, min_x, max_y, max_x = (
                max(0, int(row.bbox_0) - indent),
                max(0, int(row.bbox_1) - indent),
                min(original_img.shape[0], int(row.bbox_2) + indent),
                min(original_img.shape[1], int(row.bbox_3) + indent),
            )
            init_image = original_img[min_y:max_y, min_x:max_x]
            thr = skimage.filters.threshold_otsu(init_image)
            im = skimage.morphology.dilation((init_image > thr).astype(np.uint8))
            areas.append(np.sum(im))
        return areas


@dataclass
class CiliaDetector:
    """Detects and analyzes cilia structures in images."""

    output_dir: Path
    columns_to_save: Optional[List[str]] = None
    loader: ImageLoader = ImageLoader()
    processor: ImageProcessor = ImageProcessor()
    analyzer: ImageAnalyzer = ImageAnalyzer()
    cilia_color: str = os.environ.get("CILIA_COLOR", "rg")

    def process_image(
        self, img_path: Path, save_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Processes an image to detect cilia and extract features.

        Args:
            img_path: Path to the input image.
            save_features: Whether to save extracted features.

        Returns:
            Tuple containing the processed image, labeled image, and extracted properties.
        """
        bgr = self.loader.load_image(img_path)
        rg = self.extract_channel(bgr)
        preprocessed_img = self.processor.preprocess_image(
            rg, self.output_dir, img_path
        )
        label_img, props = self.analyzer.get_props(img_path, preprocessed_img, rg)

        if save_features and self.columns_to_save:
            self.save_features(props, img_path)

        return rg, label_img, props

    def extract_channel(self, bgr: np.ndarray) -> np.ndarray:
        """Extracts the specified color channel(s) from the image.

        Args:
            bgr: Input image in BGR format.

        Returns:
            Extracted grayscale image.
        """
        if self.cilia_color in {"rg", "gr", "red+green", "green+red"}:
            return bgr[:, :, 1] + bgr[:, :, 2]
        elif self.cilia_color in {"r", "red"}:
            return bgr[:, :, 2]
        elif self.cilia_color in {"g", "green"}:
            return bgr[:, :, 1]
        else:
            raise ValueError(f"Unsupported cilia color: {self.cilia_color}")

    def save_features(self, props: pd.DataFrame, img_path: Path):
        """Saves extracted features to a CSV file.

        Args:
            props: DataFrame containing extracted features.
            img_path: Path of the processed image.
        """
        filename = self.output_dir / f"features/{img_path.stem}.csv"
        filename.parent.mkdir(parents=True, exist_ok=True)
        props[self.columns_to_save].to_csv(filename, index=False)

    def save_labeled_images(self, img_path: Path, label_img: np.ndarray):
        """Saves the labeled segmentation result as an image.

        Args:
            img_path: Path of the input image.
            label_img: Labeled image.
        """
        filename = self.output_dir / f"labels/{img_path.stem}.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename.as_posix(), label_img)

    def save_bboxes(self, img_path: Path, props: pd.DataFrame):
        """Saves bounding box annotations to a text file.

        Args:
            img_path: Path of the input image.
            props: DataFrame containing bounding box coordinates.
        """
        bbox_filename = self.output_dir / f"bboxes/{img_path.stem}.txt"
        bbox_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(bbox_filename, "w") as bbox_file:
            for row in props.itertuples():
                min_y, min_x, max_y, max_x = (
                    int(row.bbox_0),
                    int(row.bbox_1),
                    int(row.bbox_2),
                    int(row.bbox_3),
                )
                w, h = max_x - min_x, max_y - min_y
                bbox_file.write(f"0 {min_x + w / 2} {min_y + h / 2} {w} {h}\n")

    def visualize_feature(
        self,
        img_path: Path,
        labeled_img: np.ndarray,
        props: pd.DataFrame,
        visualize_name: str = "label",
        second_img: Optional[np.ndarray] = None,
    ):
        """Visualizes detected features by overlaying bounding boxes and labels on the image.

        Args:
            img_path: Path to the input image.
            labeled_img: Labeled image containing detected features.
            props: DataFrame with extracted properties.
            visualize_name: Feature to visualize (e.g., 'label').
            second_img: Optional second image for side-by-side comparison.
        """
        img = self.loader.load_image(img_path)
        img[labeled_img != 0] = [255, 255, 255]

        for row in props.itertuples():
            min_y, min_x, max_y, max_x = (
                int(row.bbox_0),
                int(row.bbox_1),
                int(row.bbox_2),
                int(row.bbox_3),
            )
            cv2.rectangle(
                img, (min_x, min_y), (max_x, max_y), (0, 155, 255), thickness=3
            )

            x, y = max(50, min_x - 40), max(50, min_y - 20)
            cv2.putText(
                img,
                f"{getattr(row, visualize_name)}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                2,
                2,
            )

        if second_img is not None:
            img2 = second_img.copy()
            white_line = np.ones((img2.shape[0], 50, 3)) * 255
            img = np.hstack((img2, white_line, img))

        filename = self.output_dir / f"images_{visualize_name}/{img_path.stem}.png"
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename.as_posix(), img)
