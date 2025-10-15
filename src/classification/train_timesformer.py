import torch
from torch.utils.data import Dataset, DataLoader
from classification.TimeSformerWrapper import TimeSformerWrapper 
from classification.video_clip_utils import read_video_as_clip
import wandb

# 매핑 딕셔너리 인덱스로 바꾸기
CLASSES = [
  "collision",        # 추돌/충돌
  "sudden_stop",      # 급정지
  "lane_change",      # 차선변경
  "signal_violation", # 신호위반
  "near_miss",        # 아차/위험접촉
  "normal"            # 정상주행
]
NUM_CLASSES = len(CLASSES)
LABEL2ID = {c:i for i,c in enumerate(CLASSES)}
ID2LABEL = {i:c for c,i in LABEL2ID.items()}

def collate_hf(batch):
    clips, labels = zip(*batch)                
    return list(clips), torch.tensor(labels, dtype=torch.long)

class AccidentDatasetHF(Dataset):
    def __init__(self, video_dir, label_file, clip_len=16):
        self.video_dir = video_dir
        self.clip_len = clip_len
        with open(label_file) as f: 
            self.samples = [line.strip().split() for line in f]

    def __getitem__(self, idx):
        import os
        name, label = self.samples[idx]
        clip = read_video_as_clip(os.path.join(self.video_dir, name), clip_len = self.clip_len)
        return clip, LABEL2ID[label]
    
    def __len__(self):
        return len(self.samples)
    
def main(): 
    wandb.init(
        project="traffic_accident_timesformer",
        name="baseline-headonly",
        config={
            "model": "facebook/timesformer-base-finetuned-k400",
            "num_classes": 10,
            "clip_len": 16,
            "batch_size": 2,
            "lr": 3e-4,
            "freeze_backbone": True
        }
    )
    label_file = "/content/drive/MyDrive/TrafficAccidentSystem/Data/train_labels.txt"
    video_dir  = "/content/drive/MyDrive/TrafficAccidentSystem/Data/Videodata"

    model = TimeSformerWrapper(num_classes=10)
    dataset = AccidentDatasetHF(video_dir=video_dir, label_file = label_file, clip_len=wandb.config["clip_len"])
    loader = DataLoader(
        dataset, 
        batch_size = wandb.config["batch_size"], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_hf,
        persistent_workers=True
    )

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-4)
    model.train().cuda()

    wandb.watch(model, log="all", log_freq=50)

    for step, (clips, labels) in enumerate(loader): 
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        outputs = model(frames=clips, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad()

        wandb.log({
            "train/loss": loss.item(),
            "lr": optim.param_groups[0]["lr"],
            "step": step
        })

        if step % 10 ==0:
            print(f"[{step}] loss: {loss.item():.4f}")
        if step%200 == 0: 
            torch.save(model.state_dict(), "checkpoint.pt")
            wandb.save("checkpoint.pt")
    
        
if __name__=="__main__":
    main()