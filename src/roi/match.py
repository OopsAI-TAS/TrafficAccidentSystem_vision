import numpy as np, cv2
from typing import Dict, Tuple
from collections import Counter
from src.types import TrackState, FrameState

def xyxy_center(b):
    x1,y1,x2,y2 = b; return int((x1+x2)/2), int((y1+y2)/2)

def clip_xyxy(b, H, W):
    x1,y1,x2,y2 = map(int, b)
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def overlap_ratio(mask_bool: np.ndarray, box) -> float:
    H,W = mask_bool.shape
    x1,y1,x2,y2 = clip_xyxy(box, H, W)
    if x2<=x1 or y2<=y1: return 0.0
    sub = mask_bool[y1:y2, x1:x2]
    return float(sub.sum()) / float((y2-y1)*(x2-x1) + 1e-6)

def majority_label(lbl_patch: np.ndarray) -> int:
    vals = lbl_patch[lbl_patch>0].ravel()
    if vals.size == 0: return -1
    return int(Counter(vals).most_common(1)[0][0])

def match_object_to_roi(
    track: TrackState,
    fs: FrameState,
    center_weight: float = 0.6,
    thr_fused: float = 0.05,
    thr_lane_ov: float = 0.05,
    thr_road_ov: float = 0.10,
):
    """트랙 1개에 대해 road/lane 매칭 + lane_id 추정"""
    road = fs.road_mask; lane = fs.lane_mask
    H,W = fs.H, fs.W
    x1,y1,x2,y2 = clip_xyxy(track.bbox, H, W)
    cx,cy = xyxy_center(track.bbox)

    # center-hit
    c_on_road = (0<=cx<W and 0<=cy<H and bool(road[cy,cx]))
    c_on_lane = (0<=cx<W and 0<=cy<H and bool(lane[cy,cx]))
    c_scores = {"road":1.0 if c_on_road else 0.0, "lane":1.0 if c_on_lane else 0.0}

    # overlap
    o_scores = {
        "road": overlap_ratio(road, track.bbox),
        "lane": overlap_ratio(lane, track.bbox),
    }

    # fused
    fused = {k: center_weight*c_scores[k] + (1-center_weight)*o_scores[k] for k in ("road","lane")}
    roi = max(fused, key=fused.get) if max(fused.values()) >= thr_fused else "unknown"

    # lane_id (lane overlap 충분할 때 다수결)
    lane_id = -1
    if o_scores["lane"] >= thr_lane_ov and fs.lane_labels is not None:
        lbl = fs.lane_labels[y1:y2, x1:x2]
        lane_patch = lane[y1:y2, x1:x2]
        lane_id = majority_label(lbl[lane_patch])

    flags = {
        "on_road": bool(o_scores["road"] >= thr_road_ov),
        "on_lane": bool(o_scores["lane"] >= thr_lane_ov),
        "center_on_lane": c_on_lane,
    }

    # 결과 반영
    track.lane_id = lane_id
    track.flags = {**track.flags, **flags}
    return roi, flags, {"center":c_scores, "overlap":o_scores, "fused":fused}
