import astropy.units as u
import cv2
from dataclasses import dataclass
from dotenv import load_dotenv
from fil_finder import FilFinder2D
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import skimage
from typing import List, Optional, Tuple


load_dotenv("env.txt")
logger = logging.getLogger(__name__)


@dataclass
class ImageLoader:
    def load_image(self, img_path: Path) -> np.ndarray:
        img = cv2.imread(img_path.as_posix())
        if img is None:
            logger.debug(f'Image {img_path} is not found. Using black image instead.', )
            return np.zeros((512, 512, 3))
        return img


@dataclass
class ImageProcessor:
    gauss_sigma: float = float(os.environ.get("CILIA_GAUSS_SIGMA", 8))
    disk_radius: int = int(os.environ.get("CILIA_TOPHAT_RADIUS", 13))
    debug: bool = os.environ.get("DEBUG", "False") == "True"

    def preprocess_image(self, rg_image: np.ndarray, output_dir: Path, img_path: Path) -> np.ndarray:
        def save_interim_results(img: np.ndarray, dir_name: str):
            if self.debug:
                filename = output_dir / dir_name / f"{img_path.stem}.png"
                filename.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename.as_posix(), img)
        
        save_interim_results(rg_image, "0_rg")

        clahe = skimage.exposure.equalize_adapthist(rg_image)
        save_interim_results(np.uint8(clahe * 255), "1_clahe")

        blurred = skimage.filters.gaussian(clahe, sigma=self.gauss_sigma)
        save_interim_results(np.uint8(blurred * 255), "2_blurred")

        footprint = skimage.morphology.disk(self.disk_radius)
        tophat = skimage.morphology.white_tophat(blurred, footprint)
        save_interim_results(np.uint8(tophat * 255), "3_tophat")

        thresh = skimage.filters.threshold_yen(tophat)
        thresholded = (tophat > thresh).astype(np.uint8) * 255
        save_interim_results(thresholded, "4_thresholded")

        kernel = np.ones((5, 5), np.uint8)
        preprocessed_img = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=1)
        save_interim_results(preprocessed_img, "5_preprocessed_img")

        return preprocessed_img

@dataclass
class ImageAnalyzer:
    min_length: float = float(os.environ.get("CILIA_MIN_LENGTH", 14))
    min_area: float = float(os.environ.get("CILIA_MIN_AREA", 256))
    min_eccentricity: float = float(os.environ.get("CILIA_MIN_ECCENTRICITY", 0.46))
    min_perimeter: float = float(os.environ.get("CILIA_MIN_PERIMETER", 67))
    exclude_edge_cilia: bool = os.environ.get("EXCLUDE_EDGE_CILIA", "False") == "True"
    edge_margin: int = 3

    def get_props(
        self, img_path: Path, preprocessed_img: np.ndarray, original_img: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        label_img = skimage.measure.label(preprocessed_img)
        props = skimage.measure.regionprops_table(
            label_img, preprocessed_img, properties=(
                'label',
                'area',
                'orientation',
                'axis_major_length',
                'axis_minor_length',
                'bbox',
                'eccentricity',
                'image',
                'perimeter',
            )
        )
        data = pd.DataFrame(props)
        data.columns = [col.replace('-', '_') for col in data.columns]
        data = self.filter(data, original_img.shape)
        label_img, data = self.postprocess(data, label_img)
        if len(data) > 1:
            data['skeleton_length'] = self.calculate_skeleton_length(data)
            data['area'] = self.calculate_true_area(data, original_img)
            data['image_name'] = img_path.stem

        return label_img, data.round(2)

    def filter(self, data: pd.DataFrame, img_shape: Tuple[int, int]) -> pd.DataFrame:
        if self.exclude_edge_cilia:
            data = data[
                (data.bbox_0 > self.edge_margin) &
                (data.bbox_1 > self.edge_margin) &
                (data.bbox_2 < img_shape[0] - self.edge_margin) &
                (data.bbox_3 < img_shape[1] - self.edge_margin)
            ]
        return data[
            (data.axis_major_length >= self.min_length) &
            (data.eccentricity >= self.min_eccentricity) &
            (data.area >= self.min_area) &
            (data.perimeter >= self.min_perimeter)
        ]

    def postprocess(self, props: pd.DataFrame, label_img: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        mapping = {j: i for i, j in enumerate(props.label.tolist(), start=1)}
        props['old_label'] = props.label
        props['label'] = props.old_label.apply(mapping.get)
        props['form_factor'] = 4 * np.pi * props.area / props.perimeter ** 2

        float_columns = props.select_dtypes(include=[float]).columns
        props[float_columns] = props[float_columns].round(2)

        label_img = np.vectorize(mapping.get, otypes=[int])(label_img, 0)

        return label_img, props

    def calculate_skeleton_length(self, props: pd.DataFrame) -> List[float]:
        lengths = []
        for row in props.itertuples():
            im = row.image.astype(np.uint8)
            try:
                fil = FilFinder2D(im, mask=im)
                fil.preprocess_image(skip_flatten=True)
                fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
                fil.medskel(verbose=False)
                fil.analyze_skeletons(
                    skel_thresh=self.min_length * u.pix,
                    prune_criteria='length',
                    verbose=False,
                )
            except (TypeError, ValueError):
                fil.number_of_filaments = 0

            if fil.number_of_filaments == 0:
                skeleton_length = row.axis_major_length
                logger.debug(f'No filaments found for cilia {row.label}')
            else:
                end_coords = [coord for coord in fil.end_pts[0] if fil.skeleton_longpath[coord] > 0]
                if len(end_coords) > 1:
                    length_addition = (
                        fil.medial_axis_distance[end_coords[0]].value +
                        fil.medial_axis_distance[end_coords[1]].value
                    )
                    skeleton_length = sorted(fil.lengths().value)[-1] + length_addition
                else:
                    skeleton_length = row.axis_major_length
            lengths.append(skeleton_length)
        return lengths

    def calculate_true_area(self, data: pd.DataFrame, original_img: np.ndarray) -> List[int]:
        areas = []
        indent = 2
        for row in data.itertuples():
            min_y, min_x, max_y, max_x = (
                max(0, int(row.bbox_0) - indent),
                max(0, int(row.bbox_1) - indent),
                min(original_img.shape[0], int(row.bbox_2) + indent),
                min(original_img.shape[1], int(row.bbox_3) + indent)
            )
            init_image = original_img[min_y:max_y, min_x:max_x]
            thr = skimage.filters.threshold_otsu(init_image)
            im = skimage.morphology.dilation((init_image > thr).astype(np.uint8))
            areas.append(np.sum(im))
        return areas


@dataclass
class CiliaDetector:
    output_dir: Path
    columns_to_save: Optional[List[str]] = None
    loader: ImageLoader = ImageLoader()
    processor: ImageProcessor = ImageProcessor()
    analyzer: ImageAnalyzer = ImageAnalyzer()
    cilia_color: str = os.environ.get("CILIA_COLOR", "rg")

    def process_image(self, img_path: Path, save_features: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        bgr = self.loader.load_image(img_path)
        rg = self.extract_channel(bgr)
        preprocessed_img = self.processor.preprocess_image(rg, self.output_dir, img_path)
        label_img, props = self.analyzer.get_props(img_path, preprocessed_img, rg)

        if save_features and self.columns_to_save:
            self.save_features(props, img_path)

        return rg, label_img, props

    def extract_channel(self, bgr: np.ndarray) -> np.ndarray:
        if self.cilia_color in {"rg", "gr", "red+green", "green+red"}:
            return bgr[:, :, 1] + bgr[:, :, 2]
        elif self.cilia_color in {"r", "red"}:
            return bgr[:, :, 2]
        elif self.cilia_color in {"g", "green"}:
            return bgr[:, :, 1]
        else:
            raise ValueError(f"Unsupported cilia color: {self.cilia_color}")

    def save_features(self, props: pd.DataFrame, img_path: Path):
        filename = self.output_dir / f'features/{img_path.stem}.csv'
        filename.parent.mkdir(parents=True, exist_ok=True)
        props[self.columns_to_save].to_csv(filename, index=False)

    def save_labeled_images(self, img_path: Path, label_img: np.ndarray):
        filename = self.output_dir / f'labels/{img_path.stem}.png'
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename.as_posix(), label_img)

    def save_bboxes(self, img_path: Path, props: pd.DataFrame):
        bbox_filename = self.output_dir / f'bboxes/{img_path.stem}.txt'
        bbox_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(bbox_filename, 'w') as bbox_file:
            for row in props.itertuples():
                min_y, min_x, max_y, max_x = (
                    int(row.bbox_0), int(row.bbox_1), int(row.bbox_2), int(row.bbox_3)
                )
                w, h = max_x - min_x, max_y - min_y
                bbox_file.write(
                    f'0 {min_x + w/2} {min_y + h/2} {w} {h}\n'
                )

    def visualize_feature(
        self,
        img_path: Path,
        labeled_img: np.ndarray,
        props: pd.DataFrame,
        visualize_name: str = 'label',
        second_img_path: Optional[Path] = None,
    ):
        img = self.loader.load_image(img_path)
        img[labeled_img != 0] = [255, 255, 255]

        for row in props.itertuples():
            min_y, min_x, max_y, max_x = (
                int(row.bbox_0), int(row.bbox_1), int(row.bbox_2), int(row.bbox_3)
            )
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 155, 255), thickness=3)

            x, y = max(50, min_x - 40), max(50, min_y - 20)
            cv2.putText(
                img, f"{getattr(row, visualize_name)}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 2, 2,
            )

        if second_img_path:
            img2 = cv2.imread(second_img_path.as_posix()).copy()
            white_line = np.ones((img2.shape[0], 50, 3)) * 255
            img = np.hstack((img2, white_line, img))

        filename = self.output_dir / f'images_{visualize_name}/{img_path.stem}.png'
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename.as_posix(), img)
