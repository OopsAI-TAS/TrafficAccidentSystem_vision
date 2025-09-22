import argparse, json, math, cv2, numpy as np
from collections import defaultdict
from ultralytics import YOLO

INTERESTING = {0:"person", 2:"car", 3:"motorcycle", 9:"traffic_light"}

def xyxy_center(box):
    x1, y1, x2, y2 = box
    return float((x1+x2)/2), float((y1+y2)/2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml") # or strongsort.yaml
    ap.add_argument("--save_video", type=str, default="tracked.mp4")
    ap.add_argument("--save_json", type=str, default="trajectories.json")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
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

    for r in model.track(
        source=args.video, stream=True, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, verbose=False
    ):
        frame = r.orig_img.copy()
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
                if trajectories[tid]:
                    _, (px,py), _ = trajectories[tid][-1]
                    dist = math.hypot(cx-px, cy-py)
                    speed = dist * fps  # px/s
                trajectories[tid].append((frame_idx, (cx, cy), speed))

                x1,y1,x2,y2 = map(int, box)
                label = f"ID{tid} {INTERESTING[c]} {sc:.2f} v={speed:.1f}px/s"
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for _, (tx, ty), _ in trajectories[tid][-20:]:
                    cv2.circle(frame, (int(tx), int(ty)), 2, (255,255,255), -1)

        if writer is not None: writer.write(frame)
        frame_idx += 1

    if writer is not None: writer.release()

    # JSON 저장 (ROI/이벤트 감지 단계에서 재사용)
    export = {
        str(tid): [{"frame":f, "cx":cx, "cy":cy, "speed_px_s":v} for f,(cx,cy),v in seq]
        for tid, seq in trajectories.items()
    }
    with open(args.save_json, "w") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    main()