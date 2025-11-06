import argparse, json, math, cv2, numpy as np
from collections import defaultdict
from ultralytics import YOLO
from ego import VehicleBProgress
from src.ego.CameraMotion import CameraMotion
from ego.LaneChange import LaneChange
from src.events.schema import to_json_record
from src.roi.match import match_object_to_roi
from src.segmentation.YOLOPWrapper import YOLOPWrapper
from src.types import FrameState, TrackState
from utils.geometry import xyxy_center, lane_main_angle_deg
from utils.visualization import overlay_masks, to_bool, vis_roi
import json


INTERESTING = {0:"person", 2:"car", 3:"motorcycle", 9:"traffic_light"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--weights", type=str, default="yolo_trained/yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml") # or strongsort.yaml
    ap.add_argument("--save_video", type=str, default="output/tracked.mp4")
    ap.add_argument("--save_json", type=str, default="output/trajectories.json")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--roi", type=str, default="yolop", choices=["none","yolop"])
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

    yolov8 = YOLO(args.weights)
    yolop = YOLOPWrapper() if args.roi=="yolop" else None

    cam_engine = CameraMotion(fps=fps, ema=0.6, trans_thr=3.5, rot_thr_deg=1.2, cool=10)
    lane_engine = LaneChange(persist=8, stable_min=3, cool=20)
    bprog_engine = VehicleBProgress(fps=30, tau_deg = 12, v_stope=0.15, dv_start=0.3, n_stop=10, n_start=6, cooldown=20)

    per_track_events = defaultdict(list)   # tid -> [events]
    global_events = []                     # camera_shake 등

    trajectories = defaultdict(list)
    frame_idx = 0
    all_records = []

    for r in yolov8.track(
        source=args.video, stream=True, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, verbose=False
    ):
        frame = r.orig_img.copy()

        shake_evt, cam_state = cam_engine.update(frame_idx, frame)
        if shake_evt:
            global_events.append(shake_evt)
            # 화면에 작은 표시(선택)
            cv2.putText(frame, "CAMERA SHAKE!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)


        # YOLOP ROI
        roi_masks = yolop(frame) if yolop is not None else {"road":None, "lane":None}
        road = roi_masks.get("road", None)
        lane = roi_masks.get("lane", None)

        road = to_bool(frame, road)
        lane = to_bool(frame, lane)

        lc_evt = lane_engine.update(frame_idx, lane)

        phi = lane_main_angle_deg(lane)
        pred = bprog_engine.update(frame_idx, cam_state, phi, lane_change_evt=lc_evt)

        frame_vis = overlay_masks(frame, road, lane, alpha=0.35) if yolop else frame

        _, lane_labels = cv2.connectedComponents((roi_masks.get("lane").astype(np.uint8)) if lane is not None else np.zeros((H,W), np.uint8))
        # Framestate
        fs = FrameState(idx=frame_idx, H=H, W=W, 
                        road_mask = road,
                        lane_mask = lane,
                        lane_labels=lane_labels,
                        signal_state=None)
        
        tracks=[]
        id2bbox = {}
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids  = r.boxes.id
            ids  = ids.cpu().numpy().astype(int) if ids is not None else np.array([-1]*len(xyxy))

            for box, c, sc, tid in zip(xyxy, clss, conf, ids):
                if tid < 0 or c not in INTERESTING: continue
                cx, cy = xyxy_center(box)
                prev = trajectories[tid][-1] if trajectories[tid] else None
                speed = (math.hypot(cx-prev[1][0], cy-prev[1][1]) * fps) if prev else 0.0
                trajectories[tid].append((frame_idx, (cx, cy), speed))

                t = TrackState(tid=tid, cls=INTERESTING[c], bbox=box, center=(cx, cy),
                               speed=speed, heading=0.0)
                tracks.append(t)
                id2bbox[tid] = box
        
        lane_changes = lc_evt or []
        for e in lane_changes:
            per_track_events[e["id"]].append(e)
            # 시각화(선택)
            box = id2bbox.get(e["id"])
            if box is not None: 
                x1,y1,x2,y2 = map(int, box)
                cv2.putText(frame_vis, f"Lane Change {e['from']}->{e['to']}", (x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                break

        for t in tracks:
            records = to_json_record(fs.idx, t, per_track_events.get(t.tid, []))
            all_records.append(records)
            
        if writer is not None: 
          writer.write(frame_vis)
        frame_idx += 1

    if writer is not None: 
        writer.release()

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    main()
    # import torch; print("GPU is", "available" if torch.cuda.is_available() else "not available")
