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

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        pool1_output = self.pool(relu1_output)
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

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


def run_training(device_name, epochs=3, batch_size=64, data_root=ROOT / "src/datasets", limit=None):
    seed_everything()
    device = select_device(device_name)
    settings = argparse.Namespace(data_root=data_root, batch_size=batch_size, limit=limit)
    transform = transforms.ToTensor()
    train_loader, test_loader = make_loaders(datasets.MNIST, transform, transform, device, settings)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    _, total = train_model(model, train_loader, test_loader, device, epochs, optimizer)
    return total


def main_gpu(epochs=3, batch_size=64, data_root=ROOT / "src/datasets", limit=None):
    """모델과 각 배치를 CUDA에 배치하여 MNIST를 학습·평가하고 시간을 반환."""
    return run_training("cuda", epochs, batch_size, data_root, limit)


def main_cpu(epochs=3, batch_size=64, data_root=ROOT / "src/datasets", limit=None):
    return run_training("cpu", epochs, batch_size, data_root, limit)


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN GPU 학습 (main_gpu)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compare", action="store_true", help="동일 조건에서 CPU/GPU 학습 시간 비교")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--data-root", type=Path, default=ROOT / "src/datasets")
    parser.add_argument("--limit", type=positive_int, help="실행 확인용 최대 학습/평가 샘플 수")
    args = parser.parse_args()
    params = dict(epochs=args.epochs, batch_size=args.batch_size, data_root=args.data_root, limit=args.limit)
    if args.compare:
        select_device("cuda")
        cpu_time = main_cpu(**params)
        gpu_time = main_gpu(**params)
        print(f"CPU {cpu_time:.2f}s | GPU {gpu_time:.2f}s | Speedup {cpu_time / gpu_time:.2f}x")
    elif args.device == "cuda":
        main_gpu(**params)
    else:
        main_cpu(**params)


if __name__ == "__main__":
    main()
