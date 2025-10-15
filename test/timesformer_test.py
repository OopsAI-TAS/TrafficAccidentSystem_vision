# dry_train.py
import os, torch
from torch.utils.data import Dataset, DataLoader
from src.classification.TimeSformerWrapper import TimeSformerWrapper
from src.classification.video_clip_utils import read_video_as_clip

# 라벨 없이 비디오만 읽는 더미 Dataset
class UnlabeledVideoDataset(Dataset):
    def __init__(self, video_dir, clip_len=16, max_files=8):
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4",".mov",".mkv"))]
        if max_files: self.files = self.files[:max_files]

    def __getitem__(self, idx):
        import os
        path = os.path.join(self.video_dir, self.files[idx])
        clip = read_video_as_clip(path, clip_len=self.clip_len)
        return clip, -1  # dummy label

    def __len__(self):
        return len(self.files)

def collate_hf(batch):
    clips, labels = zip(*batch)
    return list(clips), torch.tensor(labels)

def main():
    video_dir = "/content/drive/MyDrive/TrafficAccidentSystem/Data/Videodata"
    NUM_CLASSES = 6  # 네 클래스 수로 맞추자
    model = TimeSformerWrapper(num_classes=NUM_CLASSES).eval().cuda()

    ds = UnlabeledVideoDataset(video_dir, clip_len=16, max_files=4)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_hf)

    with torch.no_grad():
        for clips, _ in dl:
            out = model(frames=clips)                      # labels=None → loss 없이 logits만
            logits = out.logits                            # (B, NUM_CLASSES)
            print("logits shape:", tuple(logits.shape))
            print("logits std:", float(logits.std()))      # 숫자만 나와도 forward 정상

if __name__ == "__main__":
    main()
