import numpy as np

def to_bool(img, m):
    H, W = img.shape[:2]
    
    if m is None: return None
    assert m.shape == (H, W), f"mask shape mismatch: {m.shape} vs {(H,W)}"
    return m.astype(bool)

def overlay_masks(img, road, lane, alpha=0.5):
    """
    img   : BGR uint8 (H, W, 3)
    masks : {"road": bool(H,W) or uint8(0/1), "lane": bool(H,W) 
    """
    assert img.ndim == 3 and img.shape[2] == 3, f"img shape invalid: {img.shape}"
    
    color = np.zeros_like(img, dtype=np.uint8)
    if road is not None:
        color[road] = (0, 255, 0)   # BGR
    if lane is not None:
        color[lane] = (0, 0, 255)   # BGR

    mask_region = (color[:,:,0] | color[:,:,1] | color[:,:,2]).astype(bool)
    if not np.any(mask_region):
        return img  # 칠할 곳 없으면 원본 반환

    out = img.copy()
    blend = (img[mask_region].astype(np.float32) * (1.0 - alpha) +
             color[mask_region].astype(np.float32) * alpha)
    out[mask_region] = np.clip(blend, 0, 255).astype(np.uint8)
    return out

def vis_roi(frame, road_mask, lane_mask, box, on_road, on_lane):
    import cv2, numpy as np
    over = frame.copy()
    if road_mask is not None:
        over[road_mask.astype(bool)] = (50,180,50)     # road BGR
    if lane_mask is not None:
        over[lane_mask.astype(bool)] = (180,50,50)     # lane BGR
    vis = cv2.addWeighted(frame, 0.65, over, 0.35, 0)

    x1,y1,x2,y2 = map(int, box)
    cx = int((x1+x2)/2); fy = int(y2)
    cv2.circle(vis, (cx, fy), 5, (0,255,0) if on_road else (0,0,255), -1)
    cv2.circle(vis, (cx, fy-8), 4, (255,0,0) if on_lane else (0,165,255), -1)
    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,255,255), 1)
    cv2.putText(vis, f"road={on_road} lane={on_lane}", (x1, max(0,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return vis
