import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PlaceDataset import PlaceDataset
import wandb
from tqdm.auto import tqdm


DATA_ROOT = "/content/drive/MyDrive/TrafficAccidentSystem/Data/Imagedata_split"
BATCH_SIZE = 32
EPOCHS = 20
VAL_RATIO = 0.2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_IDX = {
    "straight_road": 0,                      # 직선 도로
    "intersection_no_signal":1 ,             # 사거리 교차로(신호등 없음)
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

def build_model_class_mapping(train_root: str):
    train_root = Path(train_root)
    present_classes = []

    for cls_name in CLASS_IDX.keys():
        cls_dir = train_root / cls_name
        if not cls_dir.is_dir():
            continue

        n_imgs = sum(
            1
            for p in cls_dir.rglob("*")
            if p.suffix.lower()==".jpg"
        )
        if n_imgs > 0:
            present_classes.append(cls_name)

    # 모델 인덱스: 0 ~ (C-1)
    model_class_to_idx = {cls: i for i, cls in enumerate(present_classes)}
    idx_to_place_code = {model_idx: CLASS_IDX[cls] for cls, model_idx in model_class_to_idx.items()}

    print("모델에 사용할 클래스들:", present_classes)
    print("모델 클래스 수:", len(model_class_to_idx))

    return model_class_to_idx, idx_to_place_code

def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model_class_to_idx, idx_to_place_code = build_model_class_mapping(f"{DATA_ROOT}/train")

    train_dataset = PlaceDataset(
        root=f"{DATA_ROOT}/train",
        transform=train_tf, 
        class_to_model_idx=model_class_to_idx
    )
    val_dataset = PlaceDataset(
        root=f"{DATA_ROOT}/val",
        transform=val_tf,
        class_to_model_idx=model_class_to_idx
    )

    num_classes = len(model_class_to_idx)
    print("클래스 개수:", num_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, num_classes, idx_to_place_code, model_class_to_idx

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", ncols=120)

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}"})

    wandb.log({
        "train/loss": avg_loss,
        "train/acc": accuracy,
        "epoch": epoch
    })

    print(f"[Train] Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
    return avg_loss, accuracy


def evaluate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"[Val] Epoch {epoch}", ncols=120)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            avg_loss = total_loss / total
            acc = correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

    avg_loss = total_loss / total
    accuracy = correct / total

    wandb.log({
        "val/loss": avg_loss,
        "val/acc": accuracy,
        "epoch": epoch
    })

    print(f"[Val] Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
    return avg_loss, accuracy


def main():
    wandb.init(
        project="traffic_accident_system",
        name="resnet18_baseline",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "val_ratio": VAL_RATIO,
        }
    )

    train_loader, val_loader, num_classes, idx_to_place_code, model_class_to_idx = get_dataloaders()

    model = build_model(num_classes).to(DEVICE)
    if os.path.exists("/content/drive/MyDrive/TrafficAccidentSystem/TrafficAccidentSystem/ckpts/best_resnet18_place.pth"):
        model.load_state_dict(torch.load("/content/drive/MyDrive/TrafficAccidentSystem/TrafficAccidentSystem/ckpts/best_resnet18_place.pth"))
        print("model loaded from checkpoint.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR)

    best_acc = 0.0
    save_dir = Path("ckpts")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{EPOCHS} ==========\n")
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        _, val_acc = evaluate(model, val_loader, criterion, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_dir / "best_resnet18_place.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f" → BEST MODEL 업데이트! (val acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
