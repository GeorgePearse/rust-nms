import numpy as np
from typing import List, Tuple

def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    max_detections: int = None,
) -> np.ndarray:
    """
    Non-Maximum Suppression.

    Args:
        boxes: (N, 4) float32 array of [x1, y1, x2, y2]
        scores: (N,) float32 array of scores
        iou_threshold: IoU threshold for suppression
        max_detections: Optional limit for maximum detections

    Returns:
        (K,) array of indices to keep
    """
    ...

def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = None,
    max_detections: int = None,
    max_detections_per_class: int = None,
    parallel: bool = None,
) -> np.ndarray:
    """
    Multi-class Non-Maximum Suppression.

    Args:
        boxes: (N, 4) float32 array of [x1, y1, x2, y2]
        scores: (N,) float32 array of scores
        class_ids: (N,) int32 array of class IDs
        iou_threshold: IoU threshold for suppression
        score_threshold: Optional minimum score threshold
        max_detections: Optional limit for maximum total detections across all classes
        max_detections_per_class: Optional limit for maximum detections per class
        parallel: Optional flag to enable/disable parallel processing

    Returns:
        (K,) array of indices to keep across all classes
    """
    ...

def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression.

    Args:
        boxes: (N, 4) float32 array of [x1, y1, x2, y2]
        scores: (N,) float32 array of scores
        iou_threshold: IoU threshold for suppression
        sigma: Parameter controlling score decay (lower = more aggressive)
        score_threshold: Minimum score to keep

    Returns:
        Tuple of:
          - (K,) array of indices to keep
          - (N,) array of updated scores after soft-NMS
    """
    ...

def mask_to_polygons(
    mask: np.ndarray, threshold: float = 0.5, min_area: int = 10
) -> List[List[Tuple[float, float]]]:
    """
    Convert soft mask to polygons.

    Args:
        mask: (H, W) float32 array
        threshold: Binarization threshold
        min_area: Minimum area to keep

    Returns:
        List of polygons, each being a list of (x, y) points
    """
    ...
