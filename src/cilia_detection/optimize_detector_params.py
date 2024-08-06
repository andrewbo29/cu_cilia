import cv2
from dotenv import load_dotenv
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
import os
from pathlib import Path
from tqdm import tqdm

from cilia_detector import CiliaDetector, ImageAnalyzer, ImageProcessor
from metrics import calc_precision_recall, detection_metrics, segmentation_metrics
from utils import mask_to_bbox
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)


load_dotenv('.env') # loading mlflow credentials

def objective(trial):
    root_path = Path(__file__).parent.parent.parent.resolve()
    images_dir = root_path / 'data/CUCilia_Parameter_Range_Example/images'
    gt_dir = Path(root_path / 'data/CUCilia_Parameter_Range_Example/outlined_and_masked')
    images = sorted(list(images_dir.rglob('*.tiff')) + list(images_dir.rglob('*.tif')))

    detector = CiliaDetector(
        Path(f'/output/trial_{trial.number:06d}'),
        None,
        processor=ImageProcessor(
            gauss_sigma=trial.suggest_int('gauss_sigma', 1, 30),
            disk_radius=trial.suggest_int('disk_radius', 1, 30),
        ),
        analyzer=ImageAnalyzer(
            min_length=trial.suggest_int('min_length', 1, 30),
            min_area=trial.suggest_int('min_area', 100, 500),
            min_eccentricity=trial.suggest_float('min_eccentricity', 0, 1),
            min_perimeter=trial.suggest_float('min_perimeter', 1, 200),
        ),
        cilia_color=trial.suggest_categorical('cilia_color', ["r", "g", "rg"]),
    )

    all_seg_metrics = []
    all_det_metrics = {}
    for img_p in tqdm(sorted(images)):
        gt_path = gt_dir / f'{img_p.stem}_binary.png'
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_bboxes = mask_to_bbox(gt_mask)

        rg, label_img, props = detector.process_image(img_p, False)
        pred_mask = (label_img != 0).astype(np.uint8)
        pred_bboxes = []
        for row in props.itertuples():
            min_y, min_x, max_y, max_x = int(row.bbox_0), int(row.bbox_1), int(row.bbox_2), int(row.bbox_3)
            pred_bboxes.append([min_x, min_y, max_x, max_y])
        seg_metrics = segmentation_metrics(gt_mask, pred_mask)
        det_metrics = detection_metrics(gt_bboxes, pred_bboxes, 0.5)
        all_seg_metrics.append(seg_metrics)
        all_det_metrics[img_p.stem] = det_metrics

        detector.visualize_feature(img_p, label_img, props, visualize_name='label', second_img_path=gt_path)

    mean_seg_values = {key: np.mean([d[key] for d in all_seg_metrics if key in d]) for key in {k for d in all_seg_metrics for k in d}}
    p, r, f1 = calc_precision_recall(all_det_metrics)

    maximize_f = 0
    for max_metric in ['iou', 'f1', 'boundary_f1', 'specificity']:
        maximize_f += mean_seg_values[max_metric]
    minimize_f = 0
    for min_metric in ['fpr']:
        minimize_f += mean_seg_values[min_metric]
  
    return f1, maximize_f, minimize_f


if __name__ == '__main__':
    mlflc = MLflowCallback(
        tracking_uri='https://mlflow.apps.constr.dev',
        metric_name=['f1_det', 'max_seg', 'min_seg']
    )
    study = optuna.create_study(directions=['maximize', 'maximize', 'minimize'])
    study.optimize(objective, timeout=86400, callbacks=[mlflc])

    print('Number of finished trials: {}'.format(len(study.trials)))
    trial_with_highest_det_f1 = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with best detection F1: ")
    print(f"\tnumber: {trial_with_highest_det_f1.number}")
    print(f"\tparams: {trial_with_highest_det_f1.params}")
    print(f"\tvalues: {trial_with_highest_det_f1.values}")
    trial_with_highest_seg_max = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with best segmentation max metrics: ")
    print(f"\tnumber: {trial_with_highest_seg_max.number}")
    print(f"\tparams: {trial_with_highest_seg_max.params}")
    print(f"\tvalues: {trial_with_highest_seg_max.values}")
    trial_with_lowest_seg_min = max(study.best_trials, key=lambda t: t.values[2])
    print(f"Trial with best segmentation min metrics: ")
    print(f"\tnumber: {trial_with_lowest_seg_min.number}")
    print(f"\tparams: {trial_with_lowest_seg_min.params}")
    print(f"\tvalues: {trial_with_lowest_seg_min.values}")
