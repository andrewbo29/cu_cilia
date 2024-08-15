from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
import json
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
from rolos_sdk import Dataframe, DataStorageInterface, DataStorageType, TableColumn
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from cilia_detection.cilia_detector import CiliaDetector


load_dotenv("env.txt")
logger = logging.getLogger("cilia detection")


def process_single_image(img_p, detector, columns_to_save):
    logger.debug(img_p.name)
    rg, label_img, props = detector.process_image(img_p, True)
    detector.save_labeled_images(img_p, label_img)
    detector.save_bboxes(img_p, props)
    if np.sum(rg) == 0:
        return None
    visualizations = []
    for col in columns_to_save:
        if col == "image_name":
            continue
        try:
            visualizations.append(detector.visualize_feature(img_p, label_img, props, visualize_name=col))
        except AttributeError:
            continue
    return visualizations


def main():
    images_dir = Path(os.environ.get("IMAGES_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR")) / "cilia_results"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    vis_features = os.environ.get("CILIA_FEATURES", None)
    if vis_features is None:
        vis_features = ["area", "perimeter", "eccentricity", "form_factor", "axis_minor_length", "axis_major_length", "skeleton_length"]
    else:
        vis_features = json.loads(vis_features)
    columns_to_save = ["image_name", "label"] + vis_features
    images = sorted(list(images_dir.rglob("*.tiff")) + list(images_dir.rglob("*.tif")))
    logger.info(f"Total number of images to process: {len(images)}")
    detector = CiliaDetector(output_dir, columns_to_save)
    logger.info(f"Running cilia detection algorithm")
    
    # process images in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img_p, detector, columns_to_save) for img_p in images]
        for future in as_completed(futures):
            _ = future.result()

    # aggregate features for all cilia detected
    dfs = []
    for csv_p in (output_dir / "features").glob("*.csv"):
        df = pd.read_csv(csv_p)
        dfs.append(df)
    if len(dfs) == 0:
        return
    combined_csv = pd.concat(dfs, ignore_index=True)
    dtype_changes = {'label': int, 'area': float}
    for column, dtype in dtype_changes.items():
        if column in combined_csv.columns:
            combined_csv[column] = combined_csv[column].astype(dtype)
    combined_csv = combined_csv.round(2)
    combined_csv.to_csv(output_dir / "features.csv", index=False)

    running_locally = os.getenv("RUN_LOCALLY", "False") == "True"
    if running_locally:
        return
    table_schema = list()
    for col in combined_csv.columns:
        if col == "image_name":
            table_schema.append(TableColumn(col, str))
        elif "object" in combined_csv[col].dtype.name:
            table_schema.append(TableColumn(col, object))
        elif "int" in combined_csv[col].dtype.name:
            table_schema.append(TableColumn(col, int))
        elif "float" in combined_csv[col].dtype.name:
            table_schema.append(TableColumn(col, float))

    with DataStorageInterface.create(DataStorageType.Datacat) as storage:
        with Dataframe(
            name=f"{output_dir.parent.stem} CILIA_FEATURES", schema=table_schema, storage=storage
        ) as frame:
            frame.insert(combined_csv.values.tolist())


if __name__ == "__main__":
    main()
