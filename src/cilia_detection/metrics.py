from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
from sklearn.metrics import confusion_matrix, f1_score


def segmentation_metrics(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> Dict[str, Union[int, float]]:
    """Computes segmentation evaluation metrics.

    Args:
        gt_mask: Ground truth mask.
        pred_mask: Predicted mask.

    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, IoU, boundary F1, Hausdorff distance, specificity, and FPR.
    """
    gt_mask_bool = gt_mask > 0
    pred_mask_bool = pred_mask > 0

    gt_mask_flat = gt_mask_bool.flatten()
    pred_mask_flat = pred_mask_bool.flatten()

    tn, fp, fn, tp = confusion_matrix(gt_mask_flat, pred_mask_flat).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    iou_score = _iou(gt_mask_bool, pred_mask_bool)
    boundary_f1 = _boundary_f1_score(gt_mask_bool, pred_mask_bool)
    hausdorff = _hausdorff_distance(gt_mask_bool, pred_mask_bool)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "iou": iou_score,
        "boundary_f1": boundary_f1,
        "hausdorff_distance": hausdorff,
        "specificity": specificity,
        "fpr": fpr,
    }


def detection_metrics(
    gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float = 0.5
) -> dict:
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes."""
    if not pred_boxes:
        tp, fp, fn = 0, 0, len(gt_boxes)
    if not gt_boxes:
        tp, fp, fn = 0, len(pred_boxes), 0

    gt_idx_thr, pred_idx_thr, ious = [], [], []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = _calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    if not ious:
        tp, fp, fn = 0, len(pred_boxes), len(gt_boxes)

    else:
        args_desc = np.argsort(ious)[::-1]
        gt_match_idx, pred_match_idx = set(), set()
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            if gt_idx not in gt_match_idx and pr_idx not in pred_match_idx:
                gt_match_idx.add(gt_idx)
                pred_match_idx.add(pr_idx)

        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    acc = tp / (tp + fp + fn)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (r * p) / (r + p) if (r + p) > 0 else 0

    return {
        "true_pos": tp,
        "false_pos": fp,
        "false_neg": fn,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def calc_precision_recall(img_results: dict) -> Tuple[float, float, float]:
    """Calculates precision, recall, and F1-score from image detection results.

    Args:
        img_results: Dictionary containing results per image with keys 'true_pos', 'false_pos', 'false_neg'.

    Returns:
        Tuple containing precision, recall, and F1-score.
    """
    true_pos = sum(res["true_pos"] for res in img_results.values())
    false_pos = sum(res["false_pos"] for res in img_results.values())
    false_neg = sum(res["false_neg"] for res in img_results.values())

    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0.0

    f1 = (
        2 * (recall * precision) / (recall + precision + 1e-6)
        if recall + precision > 0
        else 0.0
    )

    return precision, recall, f1


def _iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) score between two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        IoU score.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def _boundary_f1_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Computes the F1-score for boundaries of segmentation masks.

    Args:
        gt_mask: Ground truth mask.
        pred_mask: Predicted mask.

    Returns:
        Boundary F1-score.
    """
    gt_boundary = find_boundaries(gt_mask, mode="outer")
    pred_boundary = find_boundaries(pred_mask, mode="outer")
    return f1_score(gt_boundary.flatten(), pred_boundary.flatten())


def _hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Hausdorff distance between two binary masks.

    Args:
        mask1: First binary mask.
        mask2: Second binary mask.

    Returns:
        Hausdorff distance.
    """
    mask1_points = np.argwhere(mask1)
    mask2_points = np.argwhere(mask2)
    return max(
        directed_hausdorff(mask1_points, mask2_points)[0],
        directed_hausdorff(mask2_points, mask1_points)[0],
    )


def _calc_iou_individual(pred_box: List[float], gt_box: List[float]) -> float:
    """Calculates IoU for a single predicted and ground truth bounding box.

    Args:
        pred_box: Predicted bounding box [x1, y1, x2, y2].
        gt_box: Ground truth bounding box [x1, y1, x2, y2].

    Returns:
        IoU score.
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if x1_p > x2_p or y1_p > y2_p or x1_t > x2_t or y1_t > y2_t:
        raise ValueError(f"Malformed box: pred_box={pred_box}, gt_box={gt_box}")

    if x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t:
        return 0.0

    inter_area = (min(x2_t, x2_p) - max(x1_t, x1_p) + 1) * (
        min(y2_t, y2_p) - max(y1_t, y1_p) + 1
    )
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou
