import torch 
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, TimesformerForVideoClassification


HF_MODEL_ID = "facebook/timesformer-base-finetuned-k400"

class TimeSformerWrapper(nn.Module):
    """
    입력:
      - frames: 리스트[ PIL.Image | np.ndarray(H,W,3) | torch.Tensor(3,H,W) ] 길이 = T
                또는 텐서 (B, T, C, H, W)  (float[0,1] or uint8)
    출력:
      - logits: (B, num_classes)
    """

    def __init__(self, num_classes: int, pretrained_id: str = HF_MODEL_ID, ignore_mismatch: bool = True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(pretrained_id, use_fast=True)
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=ignore_mismatch,  # head 교체 시 유용
        )
        self.freeze_backbone(except_head=True)

    def load_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)

    def freeze_backbone(self, except_head: bool = True):
        for p in self.model.parameters(): 
            p.requires_grad = False
        if except_head: 
            for p in self.model.classifier.parameters(): 
                p.requires_grad = True
        
    def forward(self, frames, labels: torch.Tensor | None=None):
        """
        frames: list[list_of_T_frames] 길이 B, 또는 텐서 (B,T,C,H,W)
                단일 샘플(list_of_T_frames)도 허용 (자동으로 배치 차원 추가)
        """

        inputs = self._prepare_inputs(frames)

        device = next(self.model.parameters()).device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device, non_blocking=True)
        if labels is not None: 
            labels = labels.to(device, non_blocking=True)

        outputs = self.model(**inputs, labels=labels)

        # outputs.logits: (B, num_classes)
        # outputs.loss: 계산된 loss
        return outputs
    
    def _prepare_inputs(self, frames):
        if isinstance(frames, torch.Tensor):
            assert frames.dim() == 5, "Tensor input must be (B, T, C, H, W)"
            b, t, c, h, w = frames.shape

            frame_list = []
            for i in range(b):
                clip = [frames[i, j].detach().cpu() for j in range(t)]
                frame_list.append(clip)

            batch = frame_list
        else: 
            batch = frames

        processed = self.processor(
            batch, 
            return_tensors="pt"
        )

        return {k: v for k, v in processed.items()}
    
    def predict_single_clip(self, clip):
        """
        clip: [frame1, frame2, ...]
            각 frame은 PIL.Image 또는 numpy array 가능
        return: (pred_idx, probs_list)
        """
        # numpy array → PIL.Image 변환 처리
        processed_clip = []
        for f in clip:
            if isinstance(f, np.ndarray):
                # numpy array는 RGB라고 가정
                processed_clip.append(Image.fromarray(f))
            else:
                # PIL.Image 그대로
                processed_clip.append(f)

        batch = [processed_clip]  # B = 1

        with torch.no_grad():
            outputs = self.forward(batch)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_idx = int(np.argmax(probs))
            
        return pred_idx, probs
