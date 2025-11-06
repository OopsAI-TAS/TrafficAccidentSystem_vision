from pathlib import Path
import torch
from torch.utils.data import Dataset
import hashlib, os, shutil
from filelock import FileLock
from video_clip_utils import read_video_as_clip

class VideoDataset(Dataset):
    def __init__(self, label_file, video_dir="/content/cache_videos", clip_len=16):
        self.video_dir = video_dir 
        self.clip_len = clip_len 
        with open(label_file) as f: 
            self.samples = [line.strip().split() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name, label = self.samples[idx]
        src_path = os.path.join(self.video_dir, video_name)

        clip = read_video_as_clip(src_path, clip_len = self.clip_len)

        return clip, int(label)
