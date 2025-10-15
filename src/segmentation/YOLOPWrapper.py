import torch, cv2, numpy as np
from torchvision import transforms as T

def letterbox(img, new_shape=(640, 384), color=(114, 114, 114)):
    """
    Resize + pad image to meet stride-multiple constraints while keeping aspect ratio.
    img: BGR numpy image
    new_shape: (W,H) target
    return: resized+padded image, ratio, (dw, dh) padding
    """
    shape = img.shape[:2]  # (H,W)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # padding (w,h)
    dw /= 2
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)

class YOLOPWrapper:
    """
    YOLOP의 forward를 감싸서 drivable area / lane line만 반환.
    - 이 래퍼는 '학습된 YOLOP 체크포인트'를 불러와
      Bx3xHxW -> dict {da_mask, ll_mask} 형태로 리턴한다고 가정.
    - 실제 모델 로딩/전처리는 네 환경의 YOLOP 구현에 맞게 수정해.
    """
    def __init__(self, device:str="cuda"):
        self.device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
        self.model = torch.hub.load("hustvl/YOLOP", "yolop", pretrained=True).to(self.device)

    @torch.no_grad()
    def __call__(self, frame):
       H, W = frame.shape[:2]
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       lb_img, r, (dw, dh) = letterbox(rgb, new_shape=(640,640))

       tf = T.Compose([
           T.ToTensor(),
           T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,0.224,0.225))
       ])
       inp = tf(lb_img).unsqueeze(0).to(self.device)
    #    print(inp.shape)

       det_out, da_out, ll_out = self.model(inp)

       # 업샘플 먼저 (640x640 패딩 크기까지)
       da_up = torch.nn.functional.interpolate(da_out, size=(lb_img.shape[0], lb_img.shape[1]),
                                                mode="bilinear", align_corners=False)
       ll_up = torch.nn.functional.interpolate(ll_out, size=(lb_img.shape[0], lb_img.shape[1]),
                                                mode="bilinear", align_corners=False)

        # 채널 차원에서 argmax → 클래스 맵 뽑기
       _, da_mask = torch.max(da_up, 1)   # da_mask: [B, H, W]
       _, ll_mask = torch.max(ll_up, 1)   # ll_mask: [B, H, W]

       da_mask = da_mask[0].cpu().numpy().astype(np.uint8)
       ll_mask = ll_mask[0].cpu().numpy().astype(np.uint8)

        # padding 제거
       top, left = int(round(dh-0.1)), int(round(dw-0.1))
       da_mask = da_mask[top: top+int(H*r), left: left+int(W*r)]
       ll_mask = ll_mask[top: top+int(H*r), left: left+int(W*r)]

        # 원본 크기로 복원
       da_mask = cv2.resize(da_mask, (W, H), interpolation=cv2.INTER_NEAREST)
       ll_mask = cv2.resize(ll_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    #    print("road pixels:", int(da_mask.sum()), "lane pixels:", int(ll_mask.sum()))

        # print(da_mask, ll_mask)
       return {"road": da_mask, "lane": ll_mask}