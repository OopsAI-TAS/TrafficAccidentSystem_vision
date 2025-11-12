import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from TimeSformerWrapper import TimeSformerWrapper 
import wandb, os
from collections import Counter
from tqdm.autonotebook import tqdm
from videodataset import VideoDataset

# 매핑 딕셔너리 인덱스로 바꾸기
# CLASSES = {
#   0: "car",    
#   1: "pedestrian",
#   2: "bicycle",
#   3: "two-wheeled vehicle",
# }

CLASSES= {
    0: "straight",
    1: "left_turn",
    2: "right_turn",
    3: "u_turn",
    4: "lane_change",
    5: "stop_or_start",
    6: "roundabout_or_intersection",
    7: "reverse_or_wrongway",
    8: "others"
}

NUM_CLASSES = len(CLASSES)
EPOCHS = 50
BATCH = 4
GRAD_ACCUM = 4
num_workers = 2
prefetch = 1

def collate_hf(batch):
    clips, labels = zip(*batch)                
    return list(clips), torch.tensor(labels, dtype=torch.long)

def make_sampler(labels):
    cnt = Counter(labels)
    weights = [1.0 / (cnt[l] ** 0.5) for l in labels]  # 1/sqrt(freq)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def evaluate(model, loader, device):
    ce = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, tot, correct = 0.0, 0, 0
    with torch.no_grad():
        for clips, labels in loader:
            labels = labels.to(device)
            outputs = model(frames=clips, labels=None)  # wrapper가 logits 반환하도록 지원해야 함
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = ce(logits, labels)
            tot_loss += loss.item(); tot += labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
    return tot_loss/tot, correct/tot
    
def main(): 
    wandb.init(
        project="traffic_accident_timesformer",
        name="vehi-head",
        config={
            "model": "facebook/timesformer-base-finetuned-k400",
            "num_classes": NUM_CLASSES,
            "clip_len": 16,
            "batch_size": BATCH,
            "lr": 3e-4,
            "freeze_backbone": True
        }
    )
    split_dir = "/content/drive/MyDrive/TrafficAccidentSystem/Data"

    train_set = VideoDataset(label_file=os.path.join(split_dir,"train_vehi.txt"))
    val_set   = VideoDataset(label_file=os.path.join(split_dir,"val_vehi.txt"))
    
    # after creating train_set / val_set
    print(f"[check] train samples: {len(train_set)}  |  val samples: {len(val_set)}")
    assert len(train_set) > 0, "Train set is empty. Check split path/labels mapping."
    assert len(val_set) > 0, "Val set is empty. Check split path/labels mapping."

    # 클래스 분포 로그
    from collections import Counter
    tr_cnt = Counter([y for _, y in train_set.samples])
    va_cnt = Counter([y for _, y in val_set.samples])
    print("[dist] train:", {k: v for k,v in tr_cnt.items()})
    print("[dist] val  :", {k: v for k,v in va_cnt.items()})


    # Sampler (불균형 보정)
    train_labels = [y for _, y in train_set.samples]
    sampler = make_sampler(train_labels)
    
    train_loader = DataLoader(
        train_set, batch_size=wandb.config["batch_size"],
        sampler=sampler, num_workers=2, prefetch_factor=prefetch,
        pin_memory=True,
        collate_fn=collate_hf, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=wandb.config["batch_size"],
        shuffle=False, num_workers=2, prefetch_factor=prefetch, 
        pin_memory=True,
        collate_fn=collate_hf, persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSformerWrapper(num_classes=NUM_CLASSES).to(device)
    # 헤드(클래스 분류기)만 확실히 최적화
    head_params = list(model.model.classifier.parameters())
    optim = torch.optim.AdamW(head_params, lr=wandb.config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    if os.path.exists("/content/drive/MyDrive/TrafficAccidentSystem/TrafficAccidentSystem/best_vehi.ckpt") and ('best_val' in torch.load("best_vehi.ckpt")):
        checkpoint = torch.load("best_vehi.ckpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint.get('best_val', float('inf'))
        global_step = checkpoint['global_step']
        print(f"✅ Checkpoint loaded. Resume from epoch {start_epoch}, best_val={best_val:.4f}")
    else: 
        best_val = float("inf")
        start_epoch = 0
        global_step = 0
    skipped = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc = f"Epoch {epoch+1}/{EPOCHS}",
                    dynamic_ncols = True, miniters = 1, smoothing = 0.1)

        optim.zero_grad(set_to_none=True)
        for step, (clips, labels) in pbar: 
            labels = labels.to(device)
            try: 
                with torch.cuda.amp.autocast():
                    out = model(frames=clips, labels=labels)  # wrapper가 loss 계산 해주면 사용
                    loss = out.loss if hasattr(out, "loss") else nn.CrossEntropyLoss()(out.logits, labels)
            except Exception as e:
                skipped += 1
                print(f"[WARN] step {step} skipped: {e}")
                continue

            scaler.scale(loss).backward()
            if (step+1) % GRAD_ACCUM == 0 or (step+1) == len(train_loader): 
                scaler.step(optim); scaler.update(); optim.zero_grad(); 
            global_step += 1
            pbar.set_postfix(step = step, loss = f"{loss.item(): .4f}", lr = optim.param_groups[0]["lr"])

            if global_step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "epoch":epoch+1, "global_step": global_step})
            if global_step % 50 == 0: 
                wandb.log({"train/skipped_batches":skipped})

        scheduler.step()
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, device)
        wandb.log({"val/loss": val_loss, "val/acc":val_acc, "epoch": epoch+1, "global_step": global_step})
        if val_loss < best_val: 
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'best_val': best_val,
                'global_step': global_step
            }, "best_vehi.ckpt")

    
        
if __name__=="__main__":
    main()