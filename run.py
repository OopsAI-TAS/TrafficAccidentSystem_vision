import argparse, json, math, cv2, numpy as np
from collections import defaultdict
from ultralytics import YOLO
from src.classification.TimeSformerWrapper import TimeSformerWrapper
from src.classification.ResNet18PlaceWrapper import ResNet18PlaceWrapper
from src.classification.video_clip_utils import read_video_as_clip
from src.ego.VehicleBProgress import VehicleBProgress
from src.ego.CameraMotion import CameraMotion
from src.ego.LaneChange import LaneChange
from src.events.schema import to_json_record
from src.roi.match import match_object_to_roi
from src.segmentation.YOLOPWrapper import YOLOPWrapper
from src.types import FrameState, TrackState
from utils.geometry import xyxy_center, lane_main_angle_deg
from utils.visualization import overlay_masks, to_bool, put_text
import json
import os


INTERESTING = {0:"person", 2:"car", 3:"motorcycle", 9:"traffic_light"}
NUM_OBJ_CLASSES = 4
NUM_VEH_CLASSES = 9
CLIP_LEN = 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--weights", type=str, default="yolo_trained/yolov8n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml") # or strongsort.yaml
    ap.add_argument("--save_video", type=str, default=None)
    ap.add_argument("--save_json", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--roi", type=str, default="yolop", choices=["none","yolop"])
    args = ap.parse_args()

    # 비디오 이름에서 확장자를 제거한 이름 추출
    video_basename = os.path.basename(args.video)
    video_name_without_ext = os.path.splitext(video_basename)[0]
    
    # 출력 디렉토리 생성
    os.makedirs("output", exist_ok=True)
    
    # 저장 경로가 지정되지 않았을 경우 자동 생성
    if args.save_video is None:
        args.save_video = f"output/{video_name_without_ext}_tracked.mp4"
    if args.save_json is None:
        args.save_json = f"output/{video_name_without_ext}_trajectories.json"

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

    obj_model = TimeSformerWrapper(num_classes=NUM_OBJ_CLASSES)
    obj_model.load_weights("/content/drive/MyDrive/TrafficAccidentSystem/TrafficAccidentSystem/ckpts/best_obj.ckpt")
    obj_model.eval()

    veh_model = TimeSformerWrapper(num_classes=NUM_VEH_CLASSES)
    veh_model.load_weights("/content/drive/MyDrive/TrafficAccidentSystem/TrafficAccidentSystem/ckpts/best_vehi.ckpt")
    veh_model.eval()

    place_model = ResNet18PlaceWrapper(num_classes=13)
    place_model.load_weights("ckpts/best_resnet18_place.pth")
    place_model.eval()

    cam_engine = CameraMotion(fps=fps, ema=0.9, trans_thr=5.5, rot_thr_deg=2.2, cool=20)
    lane_engine = LaneChange(persist=8, cooldown=20, delta_thr=0.08)
    bprog_engine = VehicleBProgress(
        fps=30,
        tau_deg=6,        # 12 → 6 (회전을 매우 민감하게)
        v_stop=0.25,      # 0.15 → 0.25 (정지를 더 확실하게만)
        dv_start=0.15,    # 0.3 → 0.15 (출발을 쉽게)
        n_stop=6,         # 10 → 6
        n_start=4,        # 6 → 4
        cooldown=15       # 20 → 15 (차선변경 감지 빈도 증가)
    )

    per_track_events = defaultdict(list)   # tid -> [events]
    global_events = []                     # camera_shake 등

    trajectories = defaultdict(list)
    pred = {}
    frame_idx = 0
    all_records = []

    video_name = os.path.basename(args.video)  # bb_1_010806_two-wheeled-vehicle_148_275.mp4
    stem = video_name.split('.')[0]            # bb_1_010806_two-wheeled-vehicle_148_275
    parts = stem.split('_')                    # ["bb", "1", "010806", "two-wheeled-vehicle", "148", "275"]

    filming_way = parts[0]                     # "bb"
    video_date  = parts[2]                     # "010806"  (yymmdd)


    for r in yolov8.track(
        source=args.video, stream=True, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, verbose=False
    ):
        frame = r.orig_img.copy()
        H, W = frame.shape[:2]

        shake_evt, cam_state = cam_engine.update(frame_idx, frame)
        if shake_evt:
            global_events.append(shake_evt)

        # YOLOP ROI
        roi_masks = yolop(frame) if yolop is not None else {"road":None, "lane":None}
        road = to_bool(frame, roi_masks.get("road", None))
        lane = to_bool(frame, roi_masks.get("lane", None))

        lc_evt = lane_engine.update(frame_idx, lane)

        phi = lane_main_angle_deg(lane)
        pred = bprog_engine.update(frame_idx, cam_state, phi, lane_change_evt=lc_evt)

        frame_vis = overlay_masks(frame, road, lane, alpha=0.35) if yolop else frame

        if lane is not None: 
            lane_u8 = (lane.astype(np.uint8) if lane.dtype != np.uint8 else lane)
        else: 
            lane_u8 = np.zeros((H, W), np.uint8)

        _, lane_labels = cv2.connectedComponents(lane_u8)
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
                match_object_to_roi(track=t, fs=fs)

                tracks.append(t)
                id2bbox[tid] = box

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0,255,0),2)
                cv2.putText(frame_vis, f"{tid}", (x1, max(y1-10,0)), 
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,255,0), 2)
        
        vis_id = 0
        if lc_evt is not None:
            global_events.append(lc_evt)

            is_right = lc_evt["type"].endswith("_R")
            arrow = "- >" if is_right else "< -"
            txt = f"EGO Lane Change {arrow}  balance={lc_evt['balance']:+.3f}"
            put_text(frame_vis, txt, rgb=(0,165,255), vis_id=vis_id)
            vis_id +=1 
            
        if shake_evt: 
            put_text(frame_vis, txt="CAMERA SHAKE!", rgb = (0,0,255), vis_id=vis_id)

        for t in tracks:
            records = to_json_record(fs.idx, t, per_track_events.get(t.tid, []))
            all_records.append(records)
            
        if writer is not None: 
          writer.write(frame_vis)
        frame_idx += 1

    if writer is not None: 
        writer.release()
    
    clip = read_video_as_clip(args.video, clip_len= CLIP_LEN)
    obj_idx, obj_probs = obj_model.predict_single_clip(clip)
    veh_idx, veh_probs = veh_model.predict_single_clip(clip)
    
    # 장소 분류: 비디오의 첫 프레임 사용
    place_idx, place_probs = place_model.predict_video_frame(args.video, frame_idx=0)

    veh_b_idx = int(pred.get("pred"))

    results = {
        "video_name": video_name,      # ex) "A001_0001.mp4"
        "video_date": video_date,      # ex) "241116"
        "filming_way": filming_way,    # "cc" or "bb"

        "accident_object": int(obj_idx),    # best_obj.ckpt 결과 (0~3)
        "accident_place": int(place_idx),   # best_resnet18_place.pth 결과 (0~14)

        "vehicle_a_progress_info": int(veh_idx),  # best_veh.ckpt 결과 (0~8)
        "vehicle_b_progress_info": veh_b_idx,     # VehicleBProgress 결과 

        # 디버깅/후처리를 위한 raw 정보는 따로 넣어두면 좋음
        "raw": {
            "vehicle_b_progress_detail": pred,     # pred가 dict면 그대로
            "events": global_events,
        }
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__=="__main__":
    main()
    # import torch; print("GPU is", "available" if torch.cuda.is_available() else "not available")
