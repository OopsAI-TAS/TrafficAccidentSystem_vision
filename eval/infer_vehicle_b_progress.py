"""
VehicleBProgress 전용 추론 스크립트

전체 파이프라인 대신 VehicleBProgress만 추론하여 빠르게 평가용 데이터 생성
- YOLO tracking 없이 카메라 모션과 차선만 분석
- txt 파일에서 비디오 리스트 읽기
- 하나의 JSON 파일로 결과 저장
- 엔진 파라미터 튜닝 가능
"""

import argparse
import json
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# 부모 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.ego.VehicleBProgress import VehicleBProgress
from src.ego.CameraMotion import CameraMotion
from src.ego.LaneChange import LaneChange
from src.segmentation.YOLOPWrapper import YOLOPWrapper
from utils.geometry import lane_main_angle_deg
from utils.visualization import to_bool


def infer_single_video(video_path: str, yolop_wrapper, engine_params, verbose=True):
    """
    단일 비디오에 대해 VehicleBProgress 추론
    
    Args:
        video_path: 비디오 파일 경로
        yolop_wrapper: YOLOPWrapper 인스턴스 (재사용)
        engine_params: 엔진 파라미터 dict
        verbose: 진행 상황 출력 여부
    
    Returns:
        dict: {
            "video_name": str,
            "vehicle_b_progress_info": int
        }
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 비디오 열기
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 엔진 초기화 (파라미터 적용)
    cam_engine = CameraMotion(
        fps=fps, 
        ema=engine_params['cam_ema'],
        trans_thr=engine_params['cam_trans_thr'],
        rot_thr_deg=engine_params['cam_rot_thr_deg'],
        cool=engine_params['cam_cool']
    )
    lane_engine = LaneChange(
        persist=engine_params['lane_persist'],
        cooldown=engine_params['lane_cooldown'],
        delta_thr=engine_params['lane_delta_thr']
    )
    bprog_engine = VehicleBProgress(
        fps=fps,
        tau_deg=engine_params['tau_deg'],
        v_stop=engine_params['v_stop'],
        dv_start=engine_params['dv_start'],
        n_stop=engine_params['n_stop'],
        n_start=engine_params['n_start'],
        cooldown=engine_params['bprog_cooldown']
    )
    
    frame_idx = 0
    pred = {}
    
    # 프로그레스 바
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", disable=not verbose)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        H, W = frame.shape[:2]
        
        # 카메라 모션 분석
        shake_evt, cam_state = cam_engine.update(frame_idx, frame)
        
        # YOLOP으로 차선 분석
        roi_masks = yolop_wrapper(frame)
        lane = to_bool(frame, roi_masks.get("lane", None))
        
        # 차선 변경 감지
        lc_evt = lane_engine.update(frame_idx, lane)
        
        # 차선 주방향 각도
        phi = lane_main_angle_deg(lane)
        
        # VehicleBProgress 업데이트
        pred = bprog_engine.update(frame_idx, cam_state, phi, lane_change_evt=lc_evt)
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    # 최종 예측값 추출
    veh_b_idx = int(pred.get("pred", -1)) if pred else -1
    
    result = {
        "video_name": video_path.name,
        "vehicle_b_progress_info": veh_b_idx
    }
    
    return result


def load_video_list(txt_path: str, video_dir: str):
    """
    txt 파일에서 비디오 리스트 읽기
    
    Args:
        txt_path: 비디오 이름이 적힌 txt 파일 경로
        video_dir: 비디오 파일들이 있는 디렉토리
    
    Returns:
        list: 비디오 파일 경로 리스트
    """
    txt_path = Path(txt_path)
    video_dir = Path(video_dir)
    
    if not txt_path.exists():
        raise FileNotFoundError(f"txt 파일을 찾을 수 없습니다: {txt_path}")
    
    video_paths = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            video_name = line.strip()
            if video_name:  # 빈 줄 제외
                video_path = video_dir / video_name
                if video_path.exists():
                    video_paths.append(video_path)
                else:
                    print(f"⚠️  비디오를 찾을 수 없어 건너뜁니다: {video_path}")
    
    return video_paths


def infer_batch(video_paths, output_json, engine_params, yolop_wrapper=None):
    """
    여러 비디오에 대해 일괄 추론하고 하나의 JSON 파일로 저장
    
    Args:
        video_paths: 비디오 파일 경로 리스트
        output_json: 결과를 저장할 JSON 파일 경로
        engine_params: 엔진 파라미터 dict
        yolop_wrapper: YOLOPWrapper 인스턴스 (없으면 새로 생성)
    """
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # YOLOPWrapper 초기화 (한 번만)
    if yolop_wrapper is None:
        print("YOLOPWrapper 초기화 중...")
        yolop_wrapper = YOLOPWrapper()
    
    all_results = {}
    
    print(f"\n총 {len(video_paths)}개 비디오 처리 시작\n")
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] {video_path.name}")
        
        try:
            result = infer_single_video(video_path, yolop_wrapper, engine_params, verbose=True)
            
            # video_name을 키로 사용
            all_results[result['video_name']] = {
                "vehicle_b_progress_info": result['vehicle_b_progress_info']
            }
            
            print(f"✓ 완료: vehicle_b_progress_info = {result['vehicle_b_progress_info']}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            all_results[video_path.name] = {
                "vehicle_b_progress_info": -1,
                "error": str(e)
            }
    
    # 하나의 JSON 파일로 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ 전체 처리 완료!")
    print(f"  - 총 비디오 수: {len(video_paths)}")
    print(f"  - 성공: {sum(1 for v in all_results.values() if v.get('vehicle_b_progress_info', -1) >= 0)}")
    print(f"  - 실패: {sum(1 for v in all_results.values() if v.get('vehicle_b_progress_info', -1) < 0)}")
    print(f"  - 결과 저장: {output_json}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="VehicleBProgress 전용 추론 스크립트 (빠른 평가용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 입력 옵션
    parser.add_argument(
        "--video_list", type=str, required=True,
        help="비디오 이름이 적힌 txt 파일 경로"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="비디오 파일들이 있는 디렉토리"
    )
    parser.add_argument(
        "--output", type=str, default="output/predictions.json",
        help="결과를 저장할 JSON 파일 경로"
    )
    
    # VehicleBProgress 파라미터
    parser.add_argument("--tau_deg", type=float, default=12.0,
                        help="회전 감지 임계값 (도)")
    parser.add_argument("--v_stop", type=float, default=0.15,
                        help="정지 판단 속도")
    parser.add_argument("--dv_start", type=float, default=0.3,
                        help="출발 판단 속도 증가량")
    parser.add_argument("--n_stop", type=int, default=10,
                        help="정지 판단 프레임 수")
    parser.add_argument("--n_start", type=int, default=6,
                        help="출발 판단 프레임 수")
    parser.add_argument("--bprog_cooldown", type=int, default=20,
                        help="VehicleBProgress 이벤트 쿨다운 (프레임)")
    
    # CameraMotion 파라미터
    parser.add_argument("--cam_ema", type=float, default=0.9,
                        help="카메라 모션 EMA 계수")
    parser.add_argument("--cam_trans_thr", type=float, default=5.5,
                        help="카메라 translation 임계값")
    parser.add_argument("--cam_rot_thr_deg", type=float, default=2.2,
                        help="카메라 rotation 임계값 (도)")
    parser.add_argument("--cam_cool", type=int, default=20,
                        help="카메라 쿨다운 (프레임)")
    
    # LaneChange 파라미터
    parser.add_argument("--lane_persist", type=int, default=8,
                        help="차선 변경 지속 프레임")
    parser.add_argument("--lane_cooldown", type=int, default=20,
                        help="차선 변경 쿨다운 (프레임)")
    parser.add_argument("--lane_delta_thr", type=float, default=0.08,
                        help="차선 변경 감지 임계값")
    
    args = parser.parse_args()
    
    # 엔진 파라미터 수집
    engine_params = {
        'tau_deg': args.tau_deg,
        'v_stop': args.v_stop,
        'dv_start': args.dv_start,
        'n_stop': args.n_stop,
        'n_start': args.n_start,
        'bprog_cooldown': args.bprog_cooldown,
        'cam_ema': args.cam_ema,
        'cam_trans_thr': args.cam_trans_thr,
        'cam_rot_thr_deg': args.cam_rot_thr_deg,
        'cam_cool': args.cam_cool,
        'lane_persist': args.lane_persist,
        'lane_cooldown': args.lane_cooldown,
        'lane_delta_thr': args.lane_delta_thr,
    }
    
    print("="*60)
    print("VehicleBProgress 추론 시작")
    print("="*60)
    print(f"비디오 리스트: {args.video_list}")
    print(f"비디오 디렉토리: {args.video_dir}")
    print(f"출력 파일: {args.output}")
    print("\n엔진 파라미터:")
    for k, v in engine_params.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")
    
    # YOLOPWrapper 초기화 (한 번만)
    print("YOLOPWrapper 초기화 중...")
    yolop_wrapper = YOLOPWrapper()
    print("✓ 초기화 완료\n")
    
    # 비디오 경로 수집
    video_paths = load_video_list(args.video_list, args.video_dir)
    
    if len(video_paths) == 0:
        raise ValueError(f"비디오를 찾을 수 없습니다. txt 파일과 비디오 디렉토리를 확인하세요.")
    
    print(f"✓ {len(video_paths)}개 비디오 로드 완료\n")
    
    # 일괄 추론
    infer_batch(video_paths, args.output, engine_params, yolop_wrapper)


if __name__ == "__main__":
    main()
