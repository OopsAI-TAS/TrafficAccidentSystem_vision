import cv2
import numpy as np

def read_video_as_clip(path: str, clip_len: int=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = np.linspace(0, max(total-1, 0), clip_len).astype(int) if total > 0 else np.arange(clip_len)

    frames = []
    cur = 0
    ptr = 0

    while True: 
        ret, frame = cap.read()
        if not ret: break
        if cur == idx[ptr]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            ptr +=1
            if ptr >= len(idx): break
        cur +=1

    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(clip_len)]
    elif len(frames) < clip_len: 
        frames += [frames[-1]] * (clip_len-len(frames))
    
    return frames
