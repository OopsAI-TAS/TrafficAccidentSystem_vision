from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class PlaceDataset(Dataset):
    def __init__(self, root, transform=None, class_to_model_idx=None):
        self.root = Path(root)
        self.transform = transform
        self.class_to_model_idx = class_to_model_idx
        self.samples = []

        for cls_name, model_idx in self.class_to_model_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue

            for p in cls_dir.rglob("*"):
                if p.suffix.lower()==".jpg":
                    self.samples.append((p, model_idx))

        print(f"[{self.root.name}] 총 이미지 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
