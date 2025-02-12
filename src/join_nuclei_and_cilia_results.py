from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import cv2
from dotenv import load_dotenv
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from rolos_sdk import Dataframe, DataStorageInterface, DataStorageType, TableColumn
from scipy.spatial.distance import cdist
from skimage.transform import resize
from skimage.segmentation import find_boundaries
from tqdm import tqdm


load_dotenv("env.txt")
logger = logging.getLogger("postprocessing")


def get_label_img(coords, img_shape):
    label_img = np.zeros((512, 512))
    for coord in coords:
        label_img[coord] = 1
    label_img = resize(
        label_img,
        (img_shape[0], img_shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )
    return label_img


def find_boundary_coords(label_img):
    boundary = find_boundaries(label_img, mode="inner")
    return np.where(boundary)


def process_image(data, img_stem, bgr, cilia_results_dir, h, w, Rx, Ry):
    img_data = data[img_stem]
    nuclei_data = pd.DataFrame.from_dict(img_data, orient="index")
    nuclei_data["cm_yx"] = nuclei_data["center_mass"].apply(lambda coords: [coords[0] * Ry, coords[1] * Rx])

    bbox_path = cilia_results_dir / f"bboxes/{img_stem}.txt"
    with open(bbox_path, "r") as f:
        cilias = f.readlines()

    if len(cilias) == 0:
        return None

    cilias_yx = {i: (i, [float(coord) for coord in cilia.strip().split(" ")[1:3]][::-1]) for i, cilia in enumerate(cilias)}
    cilias_data = pd.DataFrame.from_dict(cilias_yx).T.drop(columns=0)
    cilias_data.columns = ["cm_yx"]
    cilias_data["image_name"] = img_stem
    cilias_data["label"] = cilias_data.index + 1
    cilias_data = cilias_data.set_index("label")

    label_img = cv2.imread((cilia_results_dir / f"labels/{img_stem}.png").as_posix(), cv2.IMREAD_UNCHANGED)
    cilias_array = np.array(cilias_data["cm_yx"].tolist(), dtype=float)
    nuclei_array = np.array(nuclei_data["cm_yx"].tolist(), dtype=float)

    cm_distances = cdist(cilias_array, nuclei_array)
    argmins = cm_distances.argmin(axis=1)
    mins = cm_distances.min(axis=1)
    cilias_data["nearest_cm_"] = argmins + 1
    cilias_data["nearest_cm_dist"] = mins
    cilias_data["nearest_cm_coord"] = nuclei_data["cm_yx"].iloc[argmins].tolist()

    intersections = np.empty((len(nuclei_data), len(cilias_data)), dtype=object)
    min_bound_dis = np.empty((len(nuclei_data), len(cilias_data)), dtype=float)
    min_bound_coord = np.empty((len(nuclei_data), len(cilias_data)), dtype=object)

    for i, nuclei_row in enumerate(nuclei_data.itertuples()):
        nuclei_label = get_label_img(nuclei_row.coords, (h, w))
        boundary_y, boundary_x = find_boundary_coords(nuclei_label)
        boundary_yx = list(zip(boundary_y, boundary_x))

        for j, cilia_row in enumerate(cilias_data.itertuples()):
            cilia_label = (label_img == cilia_row.Index)
            bound_distances = cdist(np.array(cilia_row.cm_yx).reshape(1, -1), boundary_yx)
            min_bound_dis[i, j] = bound_distances.min()
            min_bound_coord[i, j] = boundary_yx[bound_distances.argmin()]
            intersections[i, j] = np.logical_and(cilia_label, nuclei_label).any()

    cilias_data["nearest_bound_cell"] = np.argmin(min_bound_dis, axis=0)
    cilias_data["nearest_bound_dist"] = min_bound_dis[cilias_data["nearest_bound_cell"]].diagonal()
    cilias_data["nearest_bound_coord"] = min_bound_coord[cilias_data["nearest_bound_cell"]].diagonal()
    nuclei, cilias = np.where(intersections)
    cilias_data["intersection_with_cell"] = {
        element+1: nuclei[cilias == element].tolist()
        if len(nuclei[cilias == element]) != 0 else False
        for element in range(len(cilias_data))
    }

    df = pd.read_csv(cilia_results_dir / f"features/{img_stem}.csv")
    filepath = cilia_results_dir / f"features_combined/{img_stem}.csv"
    filepath.parent.mkdir(exist_ok=True, parents=True)

    combined_df = pd.merge(df, cilias_data, how="inner", on=["image_name", "label"])
    combined_df.to_csv(filepath, index=False)

    img_cilia_data = {
        "image_name": img_stem,
        "cilias_number": len(combined_df),
        "in_cells_perc": (combined_df["intersection_with_cell"] != False).sum() / len(combined_df) * 100,
        "mean_nearest_cm_dist": combined_df["nearest_cm_dist"].mean(),
        "mean_nearest_bound_dist": combined_df["nearest_bound_dist"].mean(),
    }

    overlay = bgr.copy()
    for row in cilias_data.itertuples():
        cm_yx = [int(float(i)) for i in row.cm_yx]
        nearest_cm_coord = [int(float(i)) for i in row.nearest_cm_coord]
        nearest_bound_coord = [int(float(i)) for i in row.nearest_bound_coord]

        overlay = cv2.line(overlay, cm_yx[::-1], nearest_cm_coord[::-1], (255, 255, 255), thickness=6)
        overlay = cv2.line(overlay, cm_yx[::-1], nearest_bound_coord[::-1], (0, 255, 0), thickness=6)

    bgr = cv2.addWeighted(overlay, 0.3, overlay, 0.7, 0)
    cv2.imwrite((cilia_results_dir / f"distances/{img_stem}.png").as_posix(), bgr)

    return img_cilia_data


def main():
    images_dir = Path(os.environ.get("IMAGES_DIR"))
    output_dir = Path(os.environ.get("OUTPUT_DIR"))
    cilia_results_dir = output_dir / "cilia_results"
    nuclei_results_dir = output_dir / "nuclei_results"
    df_nuclei = pd.read_csv(nuclei_results_dir / "image_nuclei_features.csv")
    logger.info("Joining results of cilia and nuclei detection")

    with open(nuclei_results_dir / "nuclei_features.pkl", "rb") as f:
        data = pickle.load(f)

    img_cilia_data = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for img_stem in data.keys():
            img_path = list(images_dir.rglob(f"{img_stem}*"))[0]
            bgr = cv2.imread(img_path.as_posix())
            h, w = bgr.shape[:2]
            old_h, old_w = 512, 512
            Rx = w / old_w
            Ry = h / old_h
            futures.append(executor.submit(process_image, data, img_stem, bgr, cilia_results_dir, h, w, Rx, Ry))

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                img_cilia_data.append(result)

    logger.info("Saving results")
    df_aggregated_cilias = pd.DataFrame(
        img_cilia_data,
        columns=["image_name", "cilias_number", "in_cells_perc", "mean_nearest_cm_dist", "mean_nearest_bound_dist"]
    )
    dff = pd.merge(df_nuclei, df_aggregated_cilias, how="inner", on=["image_name"])
    dff = dff.round(2)
    dff.to_csv((output_dir / output_dir.stem).with_suffix(".csv"))

    running_locally = os.getenv("RUN_LOCALLY", "False") == "True"
    if running_locally:
        return
    if dff.empty:
        logger.debug("No data to save")
        return
    else:
        table_schema = list()
        for col in dff.columns:
            if col == "image_name":
                table_schema.append(TableColumn(col, str))
            elif "object" in dff[col].dtype.name:
                table_schema.append(TableColumn(col, object))
            elif "int" in dff[col].dtype.name:
                table_schema.append(TableColumn(col, int))
            elif "float" in dff[col].dtype.name:
                table_schema.append(TableColumn(col, float))

        with DataStorageInterface.create(DataStorageType.Datacat) as storage:
            with Dataframe(name=output_dir.stem, schema=table_schema, storage=storage) as frame:
                frame.insert(dff.values.tolist())


if __name__ == "__main__":
    main()
