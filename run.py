import argparse, json, math, cv2, numpy as np
from collections import defaultdict
from ultralytics import YOLO
from model.ROIEngine import ROIEngine


INTERESTING = {0:"person", 2:"car", 3:"motorcycle", 9:"traffic_light"}

def xyxy_center(box):
    x1, y1, x2, y2 = box
    return float((x1+x2)/2), float((y1+y2)/2)

def bbox_mask_coverage(mask, x1, y1, x2, y2):
    """bbox 내부에서 mask(0/1)가 차지하는 비율"""
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
    """중심점이 포함된 ROI 라벨 반환 (없으면 'none')"""
    cx, cy = int(round(cx)), int(round(cy))
    labels = []
    for k, m in masks_dict.items():
        if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]:
            if m[cy, cx] > 0:
                labels.append(k)
    # 우선순위: crosswalk > sidewalk > road
    for k in ["crosswalk", "sidewalk", "road"]:
        if k in labels:
            return k
    return "none"

def overlay_masks(frame, masks, alpha=0.35):
    color_map = {
        "road":       (50, 180, 50),     # BGR
        "sidewalk":   (180, 50, 50),
        "crosswalk":  (50, 50, 180)
    }
    over = frame.copy()
    for k, m in masks.items():
        if m is None: continue
        color = color_map.get(k, (128,128,128))
        over[m.astype(bool)] = color
    return cv2.addWeighted(frame, 1-alpha, over, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml") # or strongsort.yaml
    ap.add_argument("--save_video", type=str, default="output/tracked.mp4")
    ap.add_argument("--save_json", type=str, default="output/trajectories.json")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--roi", type=str, default="deeplab", choices=["none","deeplab"])
    args = ap.parse_args()


    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError("비디오 열기 실패")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (W, H))

    model = YOLO(args.weights)
    trajectories = defaultdict(list)
    frame_idx = 0

    # Init ROI engine 
    roi_engine = None
    if getattr(args, "roi", "deeplab") != "none":
        roi_engine = ROIEngine(device="cuda")

    for r in model.track(
        source=args.video, stream=True, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, verbose=False
    ):
        frame = r.orig_img.copy()
        roi_masks = None

        if roi_engine is not None:
            roi_masks = roi_engine.predict(frame)

            # Visualization
            vis = overlay_masks(frame, roi_masks)
            frame_vis = vis.copy() 
        else:
            frame_vis = frame

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids  = r.boxes.id
            ids  = ids.cpu().numpy().astype(int) if ids is not None else np.array([-1]*len(xyxy))
            for box, c, sc, tid in zip(xyxy, clss, conf, ids):
                if tid < 0 or c not in INTERESTING: continue
                cx, cy = xyxy_center(box)
                speed = 0.0

                pos_pt = "none"
                pos_cov = "none"

                if roi_masks is not None: 
                    pos_pt = point_label_from_masks(roi_masks, cx, cy)

                    cov_scores = {}
                    for k, m in roi_masks.items():
                        cov_scores[k] = bbox_mask_coverage(m, box.x1, box.x2, box.y1, box.y2)

                    priority = ["crosswalk", "sidewalk", "road"]
                    pos_cov = max(priority, key = lambda k: (cov_scores.get(k, 0, 0), priority.index(k)))

                pos_type = pos_pt if pos_pt != "none" else pos_cov

                if trajectories[tid]:
                    _, (px,py), _ = trajectories[tid][-1]
                    dist = math.hypot(cx-px, cy-py)
                    speed = dist * fps  # px/s
                trajectories[tid].append((frame_idx, (cx, cy), speed))

                x1,y1,x2,y2 = map(int, box)
                label = f"ID{tid} {INTERESTING[c]} {sc:.2f} v={speed:.1f}px/s [{pos_type}]"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for _, (tx, ty), _ in trajectories[tid][-20:]:
                    cv2.circle(frame, (int(tx), int(ty)), 2, (255,255,255), -1)

        if writer is not None: writer.write(frame)
        frame_idx += 1

    if writer is not None: 
        writer.release()
        writer.write(frame_vis)

    # JSON 저장 (ROI/이벤트 감지 단계에서 재사용)
    export = {
        str(tid): [{"frame":f, "cx":cx, "cy":cy, "speed_px_s":v} for f,(cx,cy),v in seq]
        for tid, seq in trajectories.items()
    }
    with open(args.save_json, "w") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    main()