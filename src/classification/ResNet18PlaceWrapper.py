import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image


CLASS_IDX = {
    "straight_road": 0,                      # 직선 도로
    "intersection_no_signal": 1,             # 사거리 교차로(신호등 없음)
    "intersection_signal": 2,                # 사거리 교차로(신호등 있음)
    "t_intersection": 3,                     # T자형 교차로
    "non_road_area": 4,                      # 차도와 차도가 아닌 장소
    "parking_or_nonroad": 5,                 # 주차장(또는 차도가 아닌 장소)
    "rotary": 6,                             # 회전교차로
    "crosswalk_no_signal": 7,                # 횡단보도(신호등 없음)
    "crosswalk_signal": 8,                   # 횡단보도(신호등 있음)
    "crosswalk_none": 9,                     # 횡단보도 없음
    "crosswalk_no_signal_nearby": 10,        # 횡단보도(신호등 없음) 부근
    "crosswalk_signal_nearby": 11,           # 횡단보도(신호등 있음) 부근
    "overpass_or_underpass": 12,             # 육교 및 지하도 부근
    "highway_or_expressway": 13,             # 고속도로(자동차 전용도로 포함)
    "bicycle_road": 14,                       # 자전거 도로
}


class ResNet18PlaceWrapper:
    """ResNet18 기반 장소 분류 모델 래퍼"""
    
    def __init__(self, num_classes=15):
        """
        Args:
            num_classes: 분류할 클래스 개수 (기본 15)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # ResNet18 모델 빌드
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model = self.model.to(self.device)
        
        # 전처리 transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # 모델 인덱스 -> 장소 코드 매핑 (학습 시 생성된 순서와 동일하게 설정)
        # 실제로는 학습 시 저장된 매핑을 로드해야 함
        self.idx_to_place_code = {}
    
    def load_weights(self, checkpoint_path: str):
        """
        가중치 파일 로드
        
        Args:
            checkpoint_path: .pth 파일 경로
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"✓ ResNet18 Place 모델 로드 완료: {checkpoint_path}")
    
    def eval(self):
        """평가 모드로 전환"""
        self.model.eval()
    
    def predict_image(self, image):
        """
        단일 이미지에 대한 예측 수행
        
        Args:
            image: numpy array (BGR) 또는 PIL Image
            
        Returns:
            pred_idx: 예측된 클래스 인덱스 (모델 출력)
            probs: 각 클래스에 대한 확률 (softmax 출력)
        """
        # numpy array (BGR)를 PIL Image (RGB)로 변환
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Transform 적용
        image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        image_tensor = image_tensor.to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor)  # [1, num_classes]
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()  # [num_classes]
            pred_idx = torch.argmax(outputs, dim=1).item()
        
        return pred_idx, probs
    
    def predict_video_frame(self, video_path: str, frame_idx: int = 0):
        """
        비디오의 특정 프레임에 대한 예측 수행
        
        Args:
            video_path: 비디오 파일 경로
            frame_idx: 추출할 프레임 인덱스 (기본: 첫 프레임)
            
        Returns:
            pred_idx: 예측된 클래스 인덱스
            probs: 각 클래스에 대한 확률
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
        
        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"프레임을 읽을 수 없습니다: frame_idx={frame_idx}")
        
        return self.predict_image(frame)
    
    def get_place_code(self, model_idx: int):
        """
        모델 인덱스를 실제 장소 코드로 변환
        
        Args:
            model_idx: 모델이 예측한 인덱스
            
        Returns:
            place_code: CLASS_IDX에 정의된 실제 장소 코드 (0~14)
        """
        if self.idx_to_place_code:
            return self.idx_to_place_code.get(model_idx, -1)
        else:
            # 매핑이 없으면 모델 인덱스를 그대로 반환
            return model_idx
