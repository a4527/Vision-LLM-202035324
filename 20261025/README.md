# 2026-10-25 실습

GPU 연산과 CNN 학습, IoU·NMS 실습 코드를 네 개의 Python 파일로 정리했다. 각 파일은 노트북을 실행하지 않고도 따로 실행할 수 있다.

| 파일 | 내용 |
| --- | --- |
| `01_gpu_acceleration.py` | NumPy와 CuPy로 잔상 효과, Convolution2D와 ReLU를 실행하고 데이터 전송 시간을 포함해 CPU/GPU 성능 비교 |
| `02_mnist_gpu.py` | MNIST CNN 학습과 `main_gpu()` 구현, CPU/GPU 학습 시간 비교 |
| `03_cifar10_train.py` | CIFAR-10 데이터 증강과 정규화, CNN 학습·평가, 모델 저장 및 불러오기 |
| `04_iou_nms.py` | 좌표 변환, IoU 계산, 신뢰도 기준 필터링과 NMS 구현, OpenCV 결과와 비교 |

## 실행 환경

기본 실행에는 Python 3, NumPy, OpenCV가 필요하다. MNIST와 CIFAR-10 학습에는 서로 호환되는 PyTorch와 torchvision을 사용한다. GPU로 실행하려면 CUDA를 지원하는 GPU와 드라이버가 있어야 하며, CuPy도 CUDA 버전에 맞게 설치해야 한다. GPU 환경 설정은 노트북의 안내를 참고하면 된다.

MNIST와 CIFAR-10 데이터는 저장소의 `src/datasets`에서 읽는다. 데이터가 없으면 처음 실행할 때 torchvision이 다운로드한다.

## 실행

저장소 최상위 폴더에서 아래 명령어로 실행한다. 기본 경로는 각 스크립트의 위치를 기준으로 잡기 때문에 다른 폴더에서도 실행할 수 있다.

```bash
python3 20261025/01_gpu_acceleration.py
python3 20261025/02_mnist_gpu.py
python3 20261025/03_cifar10_train.py
python3 20261025/04_iou_nms.py
```

