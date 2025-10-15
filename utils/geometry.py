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

