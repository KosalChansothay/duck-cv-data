import numpy as np
from typing import Sequence, Dict, Any

def calculate_counting_metrics(
    gt_counts: Sequence[int | float],
    pred_counts: Sequence[int | float]
) -> Dict[str, float]:
    """
    Calculate standard crowd / object counting evaluation metrics.
    
    Args:
        gt_counts: Ground truth counts per image
        pred_counts: Predicted counts per image
        
    Returns:
        Dict containing MAE, RMSE, MRE, Pearson r, and Accuracy metrics.
    """
    gt = np.array(gt_counts, dtype=np.float64)
    pred = np.array(pred_counts, dtype=np.float64)
    
    if len(gt) == 0 or len(pred) == 0 or len(gt) != len(pred):
        raise ValueError("Ground truth and prediction sequences must be non-empty and of equal length.")
        
    diff = pred - gt
    abs_diff = np.abs(diff)
    
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    
    # Avoid zero-division in relative error
    nonzero_mask = gt > 0
    if np.any(nonzero_mask):
        mre = float(np.mean(abs_diff[nonzero_mask] / gt[nonzero_mask])) * 100.0
    else:
        mre = 0.0
        
    # Exact match accuracy (pred == gt)
    exact_match = float(np.mean(np.round(pred) == gt)) * 100.0
    
    # Within ±1 duck tolerance
    within_one = float(np.mean(abs_diff <= 1.0)) * 100.0
    
    # Pearson correlation coefficient
    if len(gt) > 1 and np.std(gt) > 0 and np.std(pred) > 0:
        pearson_r = float(np.corrcoef(gt, pred)[0, 1])
    else:
        pearson_r = 1.0 if np.all(diff == 0) else 0.0
        
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mre_percent": round(mre, 2),
        "exact_match_percent": round(exact_match, 2),
        "within_one_percent": round(within_one, 2),
        "pearson_r": round(pearson_r, 4),
        "total_evaluated_images": len(gt),
        "total_gt_ducks": int(np.sum(gt)),
        "total_pred_ducks": int(np.sum(pred))
    }
