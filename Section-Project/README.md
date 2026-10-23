# 손동작으로 조작하는 실시간 카메라 필터

카메라에 한 손을 보여주면 펼친 손가락 개수에 따라 화면 전체에 필터를 적용합니다.

| 손가락 수 | 효과 | 배운 기술 |
|---|---|---|
| 0개 | 원본 | 이미지 복사 |
| 1개 | 흑백 | BGR → Grayscale |
| 2개 | 블러 | Gaussian Blur |
| 3개 | 윤곽선 | Grayscale → Gaussian Blur → Canny |
| 4개 | 선명하게 | 샤프닝 커널과 filter2D |
| 5개 | 색상 변경 | HSV의 H값 이동 |

손이 없으면 즉시 원본으로 돌아갑니다. 손가락 수가 5프레임 연속 같을 때 필터를 바꿉니다.
화면의 `Fingers`는 현재 검출값, `Filter`는 전환 대기를 거쳐 실제 적용 중인 효과입니다.

## 필요한 환경과 설치 기준

Python 라이브러리와 모델은 `02_Computer-Vision.ipynb`의 수업 환경을 기준으로 합니다.
OpenCV와 NumPy는 기존 수업/시스템 환경의 것을 사용하며 pip로 교체하지 않습니다.
`matplotlib` 추가 설치와 Windows의 `usbipd-win` 설치는 사용자 승인 후 진행한 항목입니다.

| 항목 | 용도 | 준비 방법 |
|---|---|---|
| Python 3 | 프로그램 실행 | 수업 환경 사용. 현재 WSL 검증 버전은 3.12.3 |
| OpenCV (`cv2`) | 카메라 입력, 필터, 화면 표시 | 기존 시스템 OpenCV 또는 JetPack 제공 OpenCV |
| NumPy | 관절 각도, 커널, HSV 연산 | 기존 수업/시스템 패키지 |
| MediaPipe | 손 관절 21개 검출 | `02`의 `--no-deps` 설치 명령 |
| absl-py, flatbuffers, sounddevice | 수업의 MediaPipe 환경 구성 | `02`의 설치 명령 |
| matplotlib | 현재 MediaPipe import 시 필요 | `02`에서도 import하는 패키지. 별도 승인 후 설치 |
| hand_landmarker.task | 손 인식 모델 | `02`에 나온 공식 다운로드 주소 |
| 카메라와 화면 출력 환경 | 실시간 영상 입력 및 `imshow` 창 | WSLg 데스크톱 또는 Jetson 데스크톱 |
| GStreamer | `usb`, `csi` 모드 카메라 입력 | 해당 모드 사용 시 OpenCV의 GStreamer 지원 필요 |
| usbipd-win | Windows 카메라를 WSL로 전달 | WSL 사용 시에만 필요 |

이 프로젝트는 scikit-image, ipywidgets, pose_landmarker 모델을 사용하지 않으므로
해당 항목은 프로젝트 실행을 위해 추가 설치할 필요가 없습니다.

## Python 환경 준비

아래 Linux 명령은 모두 **저장소 최상위 폴더**에서 실행합니다.
Windows Python에 설치한 패키지와 WSL Python에 설치한 패키지는 서로 공유되지 않습니다.

### 1. 기존 OpenCV와 NumPy 확인

```bash
python3 --version
python3 -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

이 단계가 실패하면 먼저 수업의 기본 환경을 준비해야 합니다.
Jetson에서는 JetPack의 OpenCV를 사용합니다.

### 2. 가상환경 만들기와 활성화

이미 수업용 가상환경이 있다면 해당 환경을 활성화해 사용해도 됩니다.
새로 만들 때는 기존 시스템의 OpenCV와 NumPy를 사용할 수 있게 구성합니다.

```bash
python3 -m venv --system-site-packages Project01/.venv
source Project01/.venv/bin/activate
```

이미 `Project01/.venv`가 준비되어 있다면 활성화 명령만 실행합니다.
`.venv`는 Git에 포함하지 않으며, 노트북에서 Jetson으로 복사하지 않습니다.
Jetson에서는 그 장치의 수업 환경을 사용하거나 새 가상환경을 만듭니다.

### 3. MediaPipe 설치

`02`의 명령과 동일하게, MediaPipe의 의존성을 자동으로 모두 설치하지 않습니다.
이미 설치되어 있다면 생략합니다.

```bash
python3 -m pip install --no-deps mediapipe
python3 -m pip install absl-py flatbuffers sounddevice
```

Jetson의 Python 버전과 ARM64 환경에서 패키지 설치가 가능한지는 장치에서 확인해야 합니다.
설치가 실패하면 다른 버전이나 비공식 패키지를 임의로 설치하지 말고 수업 환경과 비교합니다.

### 4. matplotlib 확인

```bash
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

현재 WSL에서는 이 패키지가 없어 사용자 승인 후 설치했습니다.
**현재와 같은 NumPy 1.26.4 환경을 재현할 때만** 아래 명령을 사용합니다.
기존 NumPy와 호환되는 contourpy 버전을 함께 지정하여 NumPy가 업그레이드되는 것을 막습니다.

```bash
python3 -m pip install "matplotlib==3.11.2" "contourpy==1.3.3" "numpy==1.26.4"
```

이 버전 조합은 WSL에서 확인한 것으로, Jetson에 그대로 적용하는 공통 설치 명령이 아닙니다.
Jetson에서는 수업 환경에 이미 설치된 matplotlib을 우선 사용합니다.
별도 라이브러리가 더 필요하다는 오류가 나오면 그 내용을 먼저 확인합니다.

### 5. 손 인식 모델 준비

필요한 모델은 `hand_landmarker.task` 하나입니다. 이미 있으면 다운로드를 생략합니다.
아래는 `02`에 나온 손 인식 모델 주소입니다.

```bash
mkdir -p src/models/MediaPipe
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task
```

기본 파일 배치는 다음과 같습니다. 모델은 기본적으로 저장소의 `src` 폴더에서 읽습니다.

```text
저장소/
├── Project01/
│   ├── finger_filter.py
│   └── README.md
└── src/models/MediaPipe/
    └── hand_landmarker.task
```

다른 장치에 옮길 때는 코드와 모델을 함께 준비하세요.
모델을 다른 위치에 두었다면 `--model /경로/hand_landmarker.task`로 지정할 수 있습니다.

### 6. import 확인

```bash
python3 -c "import cv2, numpy, matplotlib, mediapipe; from mediapipe.tasks.python import vision; print('라이브러리 import 성공')"
```

## Windows 노트북 카메라를 WSL에 연결

노트북 내장 카메라를 WSL에서 쓰는 경우에만 필요합니다. Jetson에서는 생략합니다.
현재 노트북에서는 Windows의 USB 장치 목록에 내장 카메라가 표시되어 연결할 수 있었습니다.
WSL 터미널을 열어 둔 상태로 진행합니다.

### 1. Windows에 usbipd-win 설치

**Windows 관리자 PowerShell**에서 실행합니다. 이미 설치했다면 생략합니다.
이 도구는 `02`의 Python 의존성이 아니라, 별도 승인 후 사용하는 장치 연결 도구입니다.

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

설치 후 PowerShell을 다시 열고 카메라의 BUSID를 확인합니다.

```powershell
usbipd list
```

### 2. 카메라 공유 및 연결

현재 노트북의 카메라 BUSID는 `4-1`이었습니다. **다른 장치에서는 실제 목록의 값으로 변경합니다.**
`bind`는 관리자 PowerShell에서 실행하며, 장치별 최초 공유 시 필요합니다.

```powershell
usbipd bind --busid 4-1
usbipd attach --wsl --busid 4-1
```

WSL에 연결된 동안 해당 카메라는 Windows에서 사용할 수 없습니다.
WSL 재시작 등으로 연결이 해제되면 `attach` 명령을 다시 실행합니다.

### 3. WSL에서 장치와 권한 확인

```bash
ls -l /dev/video*
id
```

장치가 `root video` 소유이고 사용자에게 읽기/쓰기 권한이 없다면 실행합니다.

```bash
sudo usermod -aG video "$USER"
newgrp video
source Project01/.venv/bin/activate
```

그룹 추가는 한 번만 필요합니다. `newgrp`는 변경된 그룹을 현재 터미널에 적용합니다.
USB 장치가 붙어도 `/dev/video*`가 생기지 않으면 카메라 드라이버 상태를 추가로 확인해야 합니다.

### 4. 연결 해제

Windows PowerShell에서 실행하면 카메라를 Windows에서 다시 사용할 수 있습니다.

```powershell
usbipd detach --busid 4-1
```

연결 절차 참고: [Microsoft WSL USB 연결 안내](https://learn.microsoft.com/en-us/windows/wsl/connect-usb).

## 실행

가상환경을 활성화하고 카메라 종류에 맞는 명령 **하나**를 실행합니다.
화면이 표시되는 데스크톱 터미널에서 실행하세요.

### 노트북 WSL 또는 Jetson의 일반 USB 웹캠

```bash
python3 Project01/finger_filter.py --camera webcam --device 0
```

Linux에서는 V4L2를 직접 사용하며 MJPEG, 640×480, 30fps를 요청합니다.
실제 선택된 형식·해상도·프레임률은 시작 시 `Camera:` 줄에 표시됩니다.
현재 노트북 WSL에서 영상 출력 및 프로젝트 동작을 사용자가 확인했습니다.
다른 카메라는 MJPEG 지원 여부가 다를 수 있습니다.

### Jetson USB 카메라: 수업의 YUYV 파이프라인

```bash
python3 Project01/finger_filter.py --camera usb --device 0
```

`mp-test.py`와 동일한 YUYV(YUY2), 640×480, 30fps 파이프라인입니다.
카메라가 해당 형식을 지원하고 OpenCV에 GStreamer 지원이 있어야 합니다.
옵션을 생략하면 이 `usb` 모드가 기본값입니다.

### Jetson CSI 카메라

```bash
python3 Project01/finger_filter.py --camera csi --device 0
```

JetPack의 `nvarguscamerasrc` 파이프라인을 사용합니다.
`--device`는 이 모드에서 CSI 센서 번호이며, 요청 설정은 1280×720, 30fps입니다.
실제 센서가 이 설정을 지원해야 합니다. Jetson에서의 실제 동작은 아직 검증하지 않았습니다.

다른 카메라 번호는 `--device 2`처럼 지정합니다.
종료하려면 영상 창에서 `q` 또는 `ESC`를 누르거나 창을 닫습니다.

## 확인된 개발 환경

다음은 현재 WSL에서 확인한 버전이며 Jetson의 필수 고정 버전을 뜻하지 않습니다.

| 항목 | 버전 |
|---|---|
| WSL 커널 | 6.6.87.2-microsoft-standard-WSL2, x86_64 |
| Python | 3.12.3 |
| OpenCV | 4.6.0 (기존 시스템 패키지) |
| NumPy | 1.26.4 (기존 시스템 패키지) |
| MediaPipe | 1.1.1 |
| absl-py | 2.5.1 |
| flatbuffers | 25.12.19 |
| sounddevice | 0.5.6 |
| matplotlib | 3.11.2 |
| contourpy | 1.3.3 (matplotlib 의존성) |

라이브러리 import, 모델 로드, 빈 영상 추론을 확인했으며,
USB/IP 연결과 권한 수정 및 MJPEG 설정 후 실제 카메라 동작을 사용자가 확인했습니다.

## 오류 해결

| 증상 | 확인할 내용 |
|---|---|
| `No module named ...` | 가상환경 활성화 여부와 위 설치 단계 확인. 새 패키지를 임의로 추가하지 않기 |
| 손 인식 모델이 없음 | 모델 다운로드 및 `--model` 경로 확인 |
| `/dev/video*`가 없음 | WSL에서는 `usbipd list`와 `attach` 결과 확인 |
| 장치는 있지만 읽기/쓰기 권한이 없음 | `video` 그룹 추가 후 `newgrp video` 실행 |
| 카메라를 열 수 없음 | 올바른 `--camera` 옵션, 장치 번호, 접근 권한 확인 |
| `select() timeout` 또는 프레임 수신 실패 | `--camera webcam`으로 MJPEG 설정 시도. `Camera:` 줄과 경고 확인 |
| 장치 번호가 여러 개임 | RGB·IR 영상 및 메타데이터 장치가 함께 나타날 수 있으므로 영상 장치 번호 확인 |
| GStreamer 파이프라인 오류 | USB 모드는 YUYV 지원, CSI 모드는 JetPack/Argus와 센서 설정 확인 |
| 영상 창이 나타나지 않음 | WSLg 또는 Jetson의 데스크톱 화면에서 실행 중인지 확인 |

MediaPipe의 `inference_feedback_manager` 경고가 출력되더라도 영상이 정상적으로
처리되면 그 경고만으로 실행 실패를 의미하지는 않습니다.

## 코드 설명 순서

1. `open_camera`: 카메라 영상을 가져옵니다.
2. `main`의 손 인식: 좌우 반전한 원본을 RGB로 변환해 MediaPipe에 전달합니다. 손의 관절 좌표 21개를 받습니다.
3. `calculate_angle`: 관절 하나를 중심으로 두 벡터를 만들어 각도를 계산합니다.
4. `count_fingers`: 손가락마다 두 관절의 각도를 확인합니다. 둘 다 기준 이상이면 펴진 손가락으로 셉니다. 엄지는 150도, 나머지는 160도가 기준입니다.
5. `main`의 전환 처리: 같은 개수가 5프레임 연속 나왔을 때 필터를 선택합니다. 주먹도 이 대기 후 원본으로 바뀝니다.
6. `apply_filter`: 개수에 따라 OpenCV 필터를 적용합니다. 흑백/윤곽선 결과는 BGR로 변환해 노란 상태 글씨를 그릴 수 있게 합니다.

샤프닝은 중심 픽셀에 5를 곱하고 상하좌우 값을 빼 경계를 강조합니다.
색상 변경은 uint8 HSV에서 H값을 40만큼 이동하고 180으로 나눈 나머지를 사용합니다.
덧셈 전 int16으로 변환하여 uint8 덧셈의 넘침을 방지합니다.

## 시연 팁과 한계

- 손바닥이 카메라를 향하도록 하고 한 손 전체가 화면에 나오게 합니다.
- 손가락을 구부리거나 펼치는 자세를 명확하게 보여주세요. 특히 주먹에서는 엄지도 접어야 합니다.
- 각도에 의한 규칙 기반 판정이므로 손의 기울기, 가림, 조명에 따라 개수가 틀릴 수 있습니다.
- 색상 변경은 색이 있는 물체에서, 샤프닝은 무늬나 경계가 있는 장면에서 차이가 잘 보입니다.
- 수업에서 사용한 손 인식 모델을 활용하며, 직접 모델을 학습하지 않습니다.
