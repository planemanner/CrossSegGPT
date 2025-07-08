import numpy as np
import cv2
import math
from enum import Enum

__all__ = ['compute_iou', 'jaccard_index', 'boundary_f1_score']
"""
Caution !!!
모든 Metric 은 Callable object 형태로만 구현
"""

def compute_iou(mask1: np.ndarray, mask2: np.ndarray):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union else -1.0

def jaccard_index(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes:int):
    # mIoU
    for pred_size, gt_size in zip(pred_mask.shape, gt_mask.shape):
        assert pred_size == gt_size, "You must put the pred_mask and gt_mask having same sizes."

    ious = []
    # 0 is background class
    for cls in range(1, n_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        iou = compute_iou(pred_cls, gt_cls)

        if iou == -1.0:
            ious.append(np.nan)  # 클래스가 아예 없으면 무시
        else:
            ious.append(iou)

    return np.nanmean(ious)

def f1_measure(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes: int, eps:float=1e-6):

    # 0 is background class
    f1_score = 0
    # 하나의 이미지에 대하여 Background 만 있는 경우 대응 필요 (GT 던 PRED 던)
    
    for c in range(1, n_classes):
        tp = np.logical_and(pred_mask == c, gt_mask == c).sum()
        fp = np.logical_and(pred_mask == c, gt_mask != c).sum()
        fn = np.logical_and(pred_mask != c, gt_mask == c).sum()

        precision = tp / (fp + tp + eps)
        recall = tp / (fn + tp + eps)
        
        temp_f1 = 2 * precision * recall / (precision + recall + eps) # numerically more stable
        f1_score += temp_f1

    return f1_score / (n_classes-1)

def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def boundary_f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):

    """
    Reference : https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py

    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F

#------------------------------------------------------------------------------------------#
# GPT 가 생성한 코드이기 때문에 반드시 수정이 필요함.
#------------------------------------------------------------------------------------------#
LABEL_DIVISOR = 1000
VOID_LABEL = 255  # 무시할 픽셀 값
IOU_THRESH = 0.5

def _extract_segments(seg_map):
    """
    seg_map : (H,W) ndarray, int32
    return : dict {segment_id: {'cat': category_id, 'mask': bool ndarray}}
    """
    seg_ids = np.unique(seg_map)
    segments = {}
    for sid in seg_ids:
        if sid == VOID_LABEL:
            continue
        cat = sid // LABEL_DIVISOR
        mask = seg_map == sid
        if mask.any():
            segments[sid] = {"cat": cat, "mask": mask}
    return segments

def panoptic_quality(gt_map: np.ndarray, pred_map: np.ndarray,
                     iou_thresh: float = IOU_THRESH, label_divisor: int = LABEL_DIVISOR,
                     void_label: int = VOID_LABEL):
    """
    gt_map, pred_map : (H,W) int ndarray with segment‑ids encoded as cat*div + inst
    Returns: dict with PQ, SQ, DQ and per‑class breakdown.
    """
    global LABEL_DIVISOR
    LABEL_DIVISOR = label_divisor   # 함수 외에서도 동일 기준 사용
    
    gt_segments   = _extract_segments(gt_map)
    pred_segments = _extract_segments(pred_map)

    # 카테고리별 컨테이너
    per_class = {}

    # 1) GT–Pred IoU 매트릭스 만들기
    for g_id, g in gt_segments.items():
        cat = g["cat"]
        per_class.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0})
        for p_id, p in pred_segments.items():
            if p["cat"] != cat:
                continue
            iou = compute_iou(g["mask"], p["mask"])
            if iou >= iou_thresh:
                # IoU 계산 결과를 저장; 나중에 최대 bipartite 매칭
                per_class[cat].setdefault("matches", []).append((iou, g_id, p_id))

    # 2) 카테고리별 매칭 결정(그리디: IoU 내림차순)
    matched_gt = set()
    matched_pred = set()
    for cat, stat in per_class.items():
        matches = stat.get("matches", [])
        for iou, g_id, p_id in sorted(matches, key=lambda x: x[0], reverse=True):
            if g_id in matched_gt or p_id in matched_pred:
                continue
            matched_gt.add(g_id)
            matched_pred.add(p_id)
            stat["tp"] += 1
            stat["sum_iou"] += iou

    # 3) FP / FN 계산
    for cat, stat in per_class.items():
        gt_cat_ids   = [g for g, seg in gt_segments.items()   if seg["cat"] == cat]
        pred_cat_ids = [p for p, seg in pred_segments.items() if seg["cat"] == cat]
        stat["fn"] = len([g for g in gt_cat_ids if g not in matched_gt])
        stat["fp"] = len([p for p in pred_cat_ids if p not in matched_pred])

    # 4) PQ, SQ, DQ를 카테고리별·전체 계산
    results = {}
    pq_sum = sq_sum = dq_sum = n_cat = 0
    for cat, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        if tp == 0:
            pq = sq = dq = 0.0
        else:
            sq = s["sum_iou"] / tp
            dq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * dq
        results[cat] = {"PQ": pq, "SQ": sq, "DQ": dq, "TP": tp, "FP": fp, "FN": fn}
        pq_sum += pq
        sq_sum += sq
        dq_sum += dq
        n_cat  += 1

    results["All"] = {
        "PQ": pq_sum / n_cat if n_cat else 0.0,
        "SQ": sq_sum / n_cat if n_cat else 0.0,
        "DQ": dq_sum / n_cat if n_cat else 0.0,
        "Categories": n_cat
    }
    return results
######################## ##########################

class MetricName(Enum):
    PQ = "pq"  # Panoptic Quality
    BF1M = "bf1m" # Boundary F1 Score
    MIOU = "miou" # Mean IoU


_METRIC_FACTORY = {
    MetricName.PQ: panoptic_quality,
    MetricName.BF1M: boundary_f_measure,
    MetricName.MIOU: jaccard_index
}
