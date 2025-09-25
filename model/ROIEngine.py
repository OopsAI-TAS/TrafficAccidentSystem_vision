import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np

class ROIEngine:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device=="cuda" else "cpu"
        # torchvision deeplabv3_resnet50 (coco-stuff 기반 사전학습)
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").to(self.device).eval()
        # 표준화 변환
        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        # 클래스 인덱스 매핑(가중치에 따라 다를 수 있음!)
        # 기본값: coco-stuff 기준 추정. 필요시 직접 확인해 수정.
        self.cls_map = {
            "road":       [  3,  7, 9],  # 후보 인덱스 리스트 (여러 클래스 합집합으로 간주)
            "sidewalk":   [ 15, 16],     # 예시
            # crosswalk은 보통 명시적 클래스가 없음 → heuristic로 추정
        }

    @torch.no_grad()
    def predict(self, bgr):
        """
        입력: BGR (H,W,3) uint8
        출력: dict[str] -> mask uint8{0,1}, keys: road/sidewalk/crosswalk
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        out = self.model(x)["out"].softmax(dim=1)[0]  # (C,H,W)

        H, W = rgb.shape[:2]
        masks = {}
        for k, idx_list in self.cls_map.items():
            if len(idx_list) == 1:
                m = out[idx_list[0]]
            else:
                m = torch.max(torch.stack([out[i] for i in idx_list], dim=0), dim=0).values
            masks[k] = (m > 0.5).byte().cpu().numpy()  # threshold 0.5

        # --- crosswalk heuristic (선택): 흰색 스트라이프 패턴 + road 근처 ---
        # 간단 버전: 밝은 영역이 규칙적으로 반복되는 곳을 추정 (실험적)
        crosswalk_mask = np.zeros((H,W), dtype=np.uint8)
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # 밝은 영역 강조
            bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 31, -10)
            # 길쭉한 하얀 스트라이프 강조 (열 방향)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,9))
            stripe = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            stripe = (stripe > 0).astype(np.uint8)
            # 도로 주변에서만 인정
            road = masks.get("road", np.zeros_like(stripe))
            crosswalk_mask = (stripe & road).astype(np.uint8)
            # 노이즈 제거
            crosswalk_mask = cv2.morphologyEx(crosswalk_mask, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        except Exception:
            pass

        masks["crosswalk"] = crosswalk_mask
        return masks
