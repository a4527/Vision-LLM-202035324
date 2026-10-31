#!/usr/bin/env python3
"""03_DL-and-GPU.ipynb의 2026-10-25 실습을 독립 실행 파일로 정리."""
import argparse
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import json

class ConvBlock(nn.Module):
    """
    Convolution을 이용해 특징을 추출하고 Max Pooling을 적용하는 재사용 가능한 CNN 블록.

    기본 구조:
    Conv2d → BatchNorm2d → ReLU
    → Conv2d → BatchNorm2d → ReLU
    → MaxPool2d
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

         # Block을 구성할 Layer들을 순서대로 저장
        layers = [

            # 첫 번째 Convolution (특징 추출)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 각 출력 채널의 값 분포를 조정하여 학습 안정화
            nn.BatchNorm2d(out_channels),

            # ReLU 활성화 함수
            nn.ReLU(),

            # 두 번째 Convolution (특징 추가적으로 추출)
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            # 두 번째 Convolution의 출력을 정규화
            nn.BatchNorm2d(out_channels),

            # 두 번째 활성화 함수를 적용
            nn.ReLU(),

            # Max Pooling 적용하여 다운스케일
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
            
        ]

        # 리스트에 저장한 Layer들을 순서대로 실행하는 하나의 Sequential Module로 구성
        self.block = nn.Sequential(*layers)

    # 입력 x를 self.block의 Layer에 순서대로 전달
    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):
    """
    CIFAR-10의 32x32 RGB 이미지를 10개 클래스로 분류하는 CNN 모델.

    전체 구조:
    ConvBlock 1
    → ConvBlock 2
    → ConvBlock 3
    → Global Average Pooling
    → Dropout
    → Linear
    """

    def __init__(self):
        super().__init__()

        # 이미지에서 특징을 추출하는 부분 (ConvBlock 1 → ConvBlock 2 → ConvBlock 3)
        self.features = nn.Sequential(

            
            ConvBlock(                 # 출력: [Batch, 64, 16, 16]
                in_channels=3,
                out_channels=64,
            ),

            
            ConvBlock(                 # 출력: [Batch, 128, 8, 8]
                in_channels=64,
                out_channels=128,
            ),

            
            ConvBlock(                 # 출력: [Batch, 256, 4, 4]
                in_channels=128,
                out_channels=256,
            ),

        )

        # 각 4×4 Feature Map의 평균을 계산하여 공간 크기를 1×1로 압축
        self.global_average_pool = (
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1)     # 출력: [Batch, 256, 1, 1]
                )
        )

        # 추출된 특징을 이용해 최종 클래스를 예측
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),         # 출력: 256

            nn.Linear(                 # 출력: 10
                in_features=256,
                out_features=10,
            ),
        )

    # 입력 이미지: [Batch, 3, 32, 32]
    def forward(self, x):
        # 세 개의 ConvBlock을 통과하며 특징을 추출
        # 출력: [Batch, 256, 4, 4]
        x = self.features(x)

        # 각 채널의 4×4 값을 평균내어 1×1로 압축
        # 출력: [Batch, 256, 1, 1]
        x = self.global_average_pool(x)

        # Tensor를 2차원으로 펼침
        # [Batch, 256, 1, 1] → [Batch, 256]
        x = torch.flatten(x, start_dim=1)

        # 각 클래스에 대한 점수인 Logit을 계산
        # 출력: [Batch, 10]
        logits = self.classifier(x)

        return logits

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 모델을 학습 모드(training mode)로 변경
    model.train()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 학습 데이터를 DataLoader(train_loader)의 Batch 단위로 불러와 학습 반복
    for images, labels in data_loader:
        # 모델과 동일한 장치로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화 (이전 Batch에서 계산한 Gradient를 제거)
        optimizer.zero_grad()

        # 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
        logits = model(images)

        # Loss 계산
        loss = criterion(logits, labels)
        # 역전파를 통한 Gradient 계산
        loss.backward()
        # Gradient를 사용하여 파라미터 수정
        optimizer.step()

        # Batch 한 개의 이미지 개수
        batch_size = images.size(0)

        # 결과 누적
        total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
        predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
        correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
        sample_count += batch_size                             # 전체 샘플 수 누적

    # 평균 Loss 및 Accuracy 계산
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    # 모델을 평가 모드(evaluation mode)로 변경
    model.eval()

    # 결과를 누적하기 위한 변수
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # Gradient 계산을 비활성화 (파라미터 수정 비활성화)
    with torch.no_grad():
        # 평가 데이터를 DataLoader(test_loader)의 Batch 단위로 불러와 평가 반복
        for images, labels in data_loader:
            # 모델과 동일한 장치로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 학습된 모델의 순전파를 통해 원시 출력 저장 (클래스별 점수 10개)
            logits = model(images)

            # Loss 계산 (역전파는 수행하지 않음)
            loss = criterion(logits, labels)

            # Batch 한 개의 이미지 개수
            batch_size = images.size(0)

            # 결과 누적
            total_loss += loss.item() * batch_size                 # Loss 누적 (1개 batch의 평균 loss x batch 크기)
            predictions = logits.argmax(dim=1)                     # 가장 높은 점수를 가진 클래스를 선택
            correct_count += (predictions == labels).sum().item()  # 정답을 맞춘 개수 계산
            sample_count += batch_size                             # 전체 샘플 수 누적

        # 평균 Loss 및 Accuracy 계산
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_model(model, train_loader, test_loader, device, epochs, optimizer):
    criterion = nn.CrossEntropyLoss()
    history = []
    # 워밍업은 BatchNorm 통계나 가중치를 변경하지 않는 평가 모드로 실행.
    model.eval()
    sample = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        for _ in range(3):
            model(sample)
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        synchronize(device)
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        synchronize(device)
        elapsed = time.perf_counter() - epoch_started
        history.append(dict(train_loss=train_loss, train_accuracy=train_accuracy,
                            test_loss=test_loss, test_accuracy=test_accuracy, seconds=elapsed))
        print(f"Epoch {epoch + 1}/{epochs} | Train loss {train_loss:.4f} | "
              f"Train accuracy {train_accuracy:.2%} | Test loss {test_loss:.4f} | "
              f"Test accuracy {test_accuracy:.2%} | Time {elapsed:.2f}s", flush=True)
    synchronize(device)
    total = time.perf_counter() - started
    print(f"Total {device.type.upper()} training time: {total:.2f}s")
    return history, total


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. CPU 실행은 --device cpu를 사용하세요.")
    return torch.device(name)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return value


def make_loaders(dataset_class, train_transform, test_transform, device, args):
    train_data = dataset_class(root=str(args.data_root), train=True,
                               transform=train_transform, download=True)
    test_data = dataset_class(root=str(args.data_root), train=False,
                              transform=test_transform, download=True)
    # 빠른 실행 확인용. 기본값은 전체 데이터셋을 사용합니다.
    if args.limit:
        train_data = Subset(train_data, range(min(args.limit, len(train_data))))
        test_data = Subset(test_data, range(min(args.limit, len(test_data))))
    options = dict(num_workers=0, pin_memory=device.type == "cuda")
    return (DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **options),
            DataLoader(test_data, batch_size=1000, shuffle=False, **options))

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN 학습, 평가 및 .pth 저장")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--data-root", type=Path, default=ROOT / "src/datasets")
    parser.add_argument("--output", type=Path, default=HERE / "CIFAR10_CNN.pth")
    parser.add_argument("--limit", type=positive_int, help="실행 확인용 최대 학습/평가 샘플 수")
    parser.add_argument("--evaluate-only", action="store_true", help="--output의 저장 모델을 불러와 평가만 수행")
    args = parser.parse_args()
    seed_everything()
    device = select_device(args.device)
    model = CIFAR10CNN().to(device)
    if args.evaluate_only:
        checkpoint = torch.load(args.output, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        # 체크포인트에 기록된 학습 정규화 값을 평가에도 사용합니다.
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=checkpoint["mean"], std=checkpoint["std"])])
        test_data = datasets.CIFAR10(root=str(args.data_root), train=False,
                                     transform=transform, download=True)
        if args.limit:
            test_data = Subset(test_data, range(min(args.limit, len(test_data))))
        loader = DataLoader(test_data, batch_size=1000, num_workers=0)
        loss, accuracy = evaluate(model, loader, nn.CrossEntropyLoss(), device)
        print(f"Saved model | Test loss {loss:.4f} | Test accuracy {accuracy:.2%}")
        return
    train_loader, test_loader = make_loaders(datasets.CIFAR10, train_transform, test_transform, device, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    history, total = train_model(model, train_loader, test_loader, device, args.epochs, optimizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": history[-1]["test_accuracy"],
        "epochs": args.epochs,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "history": history,
        "total_training_seconds": total,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(
        {key: value for key, value in checkpoint.items()
         if key not in ("model_state_dict", "optimizer_state_dict")},
        indent=2), encoding="utf-8")
    # 저장 직후 새 모델에서 state_dict를 읽어 직렬화와 구조 호환성 확인.
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    CIFAR10CNN().load_state_dict(loaded["model_state_dict"])
    print(f"모델 저장 및 다시 불러오기 완료: {args.output}")


if __name__ == "__main__":
    main()
