import numpy as np
from typing import List, Tuple

def nms(
    boxes: np.ndarray, 
    scores: np.ndarray, 
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) float32 array of [x1, y1, x2, y2]
        scores: (N,) float32 array of scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        (K,) array of indices to keep
    """
    ...

def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    method: str = "linear",
    sigma: float = 0.5,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) float32 array
        scores: (N,) float32 array
        method: 'linear' or 'gaussian'
        sigma: Gaussian sigma
        iou_threshold: Linear method threshold
        score_threshold: Minimum score to keep
        
    Returns:
        Tuple of (indices, new_scores)
    """
    ...

def mask_to_polygons(
    mask: np.ndarray, 
    threshold: float = 0.5, 
    min_area: int = 10
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
