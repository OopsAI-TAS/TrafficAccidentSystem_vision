import numpy as np
import cv2, math

def xyxy_center(box):
    x1, y1, x2, y2 = box
    return float((x1+x2)/2), float((y1+y2)/2)

def bbox_mask_coverage(mask, x1, y1, x2, y2):
    if mask is None or not hasattr(mask, "shape"):
        return 0.0
    H, W = mask.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), W-1), min(int(y2), H-1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2+1, x1:x2+1]
    area_box = sub.size
    if area_box == 0:
        return 0.0
    return float(sub.sum() / area_box)

def point_label_from_masks(masks_dict, cx, cy):
    cx, cy = int(round(cx)), int(round(cy))
    labels = []
    for k, m in masks_dict.items():
        if m is None or not hasattr(m, "shape"):
            continue  # None이면 건너뛰기
        H, W = m.shape[:2]
        if 0 <= cy < H and 0 <= cx < W:
            if m[cy, cx] > 0:
                labels.append(k)
    # 우선순위
    for k in ["crosswalk", "sidewalk", "road"]:
        if k in labels:
            return k
    return "none"

def lane_main_angle_deg(lane_mask):
    """
    lane_mask: uint8 이진(0/255) 또는 0/1
    반환: φ(deg) in [-90, +90], 0이 화면 가로축(수평), 양수=반시계(왼쪽으로 기울)
    """
    if lane_mask.dtype != np.uint8:
        lane = (lane_mask>0).astype(np.uint8)*255
    else:
        lane = lane_mask.copy()

    edges = cv2.Canny(lane, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is not None and len(lines)>0:
        # angle histogram의 모드
        thetas = [theta for rho,theta in lines[:,0]]
        # θ는 법선 각 → 직선 방향은 θ-90°
        dirs = [math.degrees(t - math.pi/2) for t in thetas]
        # 중앙값이 튼튼
        phi = np.median(dirs)
        # wrap to [-90,90]
        phi = ( (phi+90) % 180 ) - 90
        return float(phi)

    # fallback: 주성분
    ys, xs = np.where(lane>0)
    if len(xs)<100:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None, maxComponents=2)
    v = eigenvectors[0]   # 주성분 방향 (x,y)
    phi = math.degrees(math.atan2(v[1], v[0]))  # [-180,180]
    # 화면 수평 기준으로 맞춤 → [-90,90]
    phi = ( (phi+90) % 180 ) - 90
    return float(phi)
