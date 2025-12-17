import numpy as np
import cv2

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

def put_text(frame_vis, txt, rgb, vis_id): 
    cv2.putText(frame_vis, txt, (20*(vis_id+1), 70*(vis_id+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rgb, 2)