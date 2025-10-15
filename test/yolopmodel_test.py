import torch
import argparse
import cv2
from src.segmentation.YOLOPWrapper import YOLOPWrapper
from run import overlay_masks
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    ap.add_argument("--save", type=str, default="output/output.jpg", help="저장할 파일 이름")
    args = ap.parse_args()

    # YOLOP segmentation 래퍼 로드
    yolop = YOLOPWrapper(device="cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 읽기
    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"이미지를 불러올 수 없음: {args.image}")

    # YOLOP 실행
    mask = yolop(frame)   # {"road": bool(H,W), "lane": bool(H,W)}

    # 마스크 overlay
    vis = frame.copy()

    print(frame.shape)
    # print(mask["road"])
    out = overlay_masks(frame, mask)

    # da_mask = mask["road"]
    # ll_mask = mask["lane"]

    # cv2.imwrite(f"road_bin.png", (da_mask.astype(np.uint8) * 255))
    # cv2.imwrite(f"lane_bin.png", (ll_mask.astype(np.uint8) * 255))

    # # (B) 색만 칠한 마스크 (배경은 검정)
    # road_only = np.zeros_like(frame)
    # lane_only = np.zeros_like(frame)
    # road_only[da_mask.astype(bool)] = (0, 255, 0)   # BGR: green
    # lane_only[ll_mask.astype(bool)] = (0, 0, 255)   # BGR: red
    # cv2.imwrite(f"road_color.png", road_only)
    # cv2.imwrite(f"lane_color.png", lane_only)

    # # (C) 두 마스크 합본 (겹치면 보라색 쪽으로 보일 수 있음)
    # combo = np.zeros_like(frame)
    # combo[da_mask.astype(bool)] = (0, 255, 0)
    # combo[ll_mask.astype(bool)] = (0, 0, 255)
    # cv2.imwrite(f"road_lane_combo.png", combo)

    # 결과 저장
    cv2.imwrite(args.save, out)
    print(f"저장 완료: {args.save}")

if __name__=="__main__":
    main()
