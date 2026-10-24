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

손을 카메라에 가까이 가져오면 필터가 강해지고, 멀리 두면 약해집니다.
화면의 `Strength`와 노란 막대로 현재 적용 강도(0~100%)를 표시합니다.
손가락 개수는 필터 종류를, 손바닥 크기는 강도를 결정합니다.
객체 추적과 칼만 필터는 사용하지 않으며, 추가 패키지나 모델은 필요하지 않습니다.

## 손 크기로 필터 강도 조절

손목(랜드마크 0번)부터 중지 시작 관절(9번)까지의 화면상 길이를 사용합니다.
손가락 끝 대신 손바닥을 기준으로 하여 손가락 개수 변화의 영향을 줄입니다.
각 좌표를 픽셀로 변환해 거리를 구한 뒤 화면 높이로 나누므로,
640×480과 1280×720 같은 입력 해상도가 바뀌어도 화면 대비 크기를 기준으로 계산합니다.

```python
palm_size = palm_length_pixels / frame_height
strength = np.clip((palm_size - far_size) / (near_size - far_size), 0, 1)
output = cv2.addWeighted(original, 1 - strength, filtered, strength, 0)
```

기본 기준은 화면 높이의 8% 이하일 때 강도 0%, 30% 이상일 때 100%입니다.
그 사이에서는 크기에 비례하여 강도가 증가합니다. 예를 들어 480픽셀 높이에서는
손바닥 길이가 약 38픽셀일 때 0%, 약 91픽셀일 때 50%, 144픽셀일 때 100%입니다.
강도 0%는 원본, 100%는 선택한 필터의 최대 효과입니다.
흑백·윤곽선도 원본과 섞어 적용하며, 블러 커널이나 Canny 임계값 자체를 바꾸지는 않습니다.
색상 변경 역시 고정된 색상 변경 결과와 원본을 혼합합니다.
주먹으로 원본 모드가 선택되거나 손이 없을 때는 강도 0%로 표시합니다.

손 크기나 카메라 화각에 따라 범위가 맞지 않으면 실행 옵션으로 조절할 수 있습니다.

```bash
python3 Section-Project/finger_filter.py --device 0 --far-size 0.06 --near-size 0.22
```

가까이 가져와도 강도가 낮으면 `--near-size`를 낮추고,
멀리 두어도 강도가 높으면 `--far-size`를 높여 보세요.
조건은 `0 <= far-size < near-size <= 1`입니다.
실제 거리를 cm로 측정하는 기능은 아니며, 손을 기울여도 강도가 변할 수 있습니다.
손바닥을 카메라 쪽으로 향하게 유지하세요.

## 필요한 실행 환경

`02_Computer-Vision.ipynb`의 YUYV 카메라 실습을 기준으로 실행합니다.
카메라 입력은 GStreamer의 `v4l2src`를 사용하며 설정은 **YUYV, 640×480, 30fps**입니다.
GStreamer에서는 YUYV 포맷을 `YUY2`라는 이름으로 지정합니다.

| 항목 | 용도 및 준비 기준 |
|---|---|
| Python 3 | 수업에서 사용하는 Python 환경 |
| OpenCV (`cv2`) | JetPack/수업 환경의 GStreamer 및 화면 출력 지원 OpenCV |
| NumPy | 수업 환경의 기존 패키지. 각도·커널·강도 계산 |
| MediaPipe | 손 관절 21개 인식. `02`의 설치 명령 사용 |
| absl-py, flatbuffers, sounddevice | `02`에서 MediaPipe와 함께 설치하는 항목 |
| matplotlib | `02`에서 사용하는 라이브러리. 현재 MediaPipe import에도 필요 |
| hand_landmarker.task | `02`에서 다운로드하는 손 인식 모델 |
| GStreamer | 카메라 파이프라인과 BGR 변환 |
| USB 카메라 | YUYV 640×480 30fps 지원 필요 |
| 데스크톱 화면 | OpenCV 영상 창을 표시할 수 있는 실행 환경 |

OpenCV와 NumPy는 기존 수업 환경의 것을 사용하고 pip로 교체하지 않습니다.
이 프로젝트에서는 scikit-image, ipywidgets, pose_landmarker 모델이 필요하지 않습니다.
다른 장치에서 만든 가상환경을 복사하지 말고 해당 장치의 수업 환경을 사용합니다.
Jetson에서의 실제 카메라 동작은 아직 검증하지 않았습니다.

## 라이브러리와 모델 준비

아래 명령은 저장소 최상위 폴더에서 **수업용 Python 환경을 활성화한 뒤** 실행합니다.

### 1. 기존 라이브러리 확인

```bash
python3 --version
python3 -c "import cv2, numpy, matplotlib; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('matplotlib:', matplotlib.__version__)"
```

누락된 기본 라이브러리가 있다면 먼저 수업의 기본 환경과 비교합니다.
GStreamer 지원 여부는 다음 출력에서 `GStreamer: YES`인지 확인합니다.

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

### 2. MediaPipe 설치

`02`의 MediaPipe 설치 명령을 사용합니다. 이미 설치되어 있으면 생략합니다.

```bash
python3 -m pip install --no-deps mediapipe
python3 -m pip install absl-py flatbuffers sounddevice
```

Jetson의 Python 버전과 ARM64 환경에서 설치가 가능한지는 장치에서 확인해야 합니다.
설치에 실패하면 다른 버전이나 비공식 패키지를 임의로 추가하지 말고 수업 환경과 비교합니다.

```bash
python3 -c "import mediapipe; from mediapipe.tasks.python import vision; print('MediaPipe import 성공')"
```

### 3. 손 인식 모델 다운로드

`02`에 나온 공식 주소를 사용합니다. 모델이 이미 있으면 생략합니다.

```bash
mkdir -p src/models/MediaPipe
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O src/models/MediaPipe/hand_landmarker.task
```

```text
저장소/
├── Section-Project/
│   ├── finger_filter.py
│   └── README.md
└── src/models/MediaPipe/
    └── hand_landmarker.task
```

코드를 다른 장치에 옮길 때 모델도 함께 준비합니다.
모델을 다른 위치에 두었다면 `--model /경로/hand_landmarker.task`로 지정합니다.

## YUYV 카메라 준비

### 1. 장치 및 지원 포맷 확인

`02`에 나온 명령으로 카메라와 지원 포맷을 확인합니다.

```bash
ls /dev/video*
sudo apt install v4l-utils
v4l2-ctl --list-formats-ext -d /dev/video0
```

`v4l-utils`가 이미 설치되어 있으면 설치 명령은 생략합니다.
출력에서 YUYV, 640×480, 30fps 지원을 확인합니다.
다른 카메라는 `/dev/video0`을 해당 장치 경로로 바꿉니다.

### 2. 수업의 GStreamer 명령으로 영상 확인

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! 'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1' ! videoconvert ! autovideosink
```

영상이 나오면 창을 닫거나 `Ctrl+C`로 테스트를 종료한 뒤 프로젝트를 실행합니다.
프로젝트 코드의 카메라 파이프라인은 `02`의 YUYV 예제와 같습니다.
`--device`로 장치 번호만 바꿀 수 있습니다.

```python
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

## 실행 방법

저장소 최상위 폴더에서 실행합니다. 기본 장치는 `/dev/video0`입니다.

```bash
python3 Section-Project/finger_filter.py
```

다른 카메라 번호를 사용하는 경우:

```bash
python3 Section-Project/finger_filter.py --device 1
```

강도 조절 범위를 변경하는 경우:

```bash
python3 Section-Project/finger_filter.py --device 0 --far-size 0.06 --near-size 0.22
```

카메라 종류를 선택하는 `--camera` 옵션은 사용하지 않습니다.
종료는 영상 창에서 `q` 또는 `ESC`를 누르거나 창을 닫으면 됩니다.

## 오류 해결과 검증 범위

| 증상 | 확인할 내용 |
|---|---|
| `No module named ...` | 수업용 Python 환경 활성화 여부와 누락된 라이브러리 확인 |
| 손 인식 모델이 없음 | 모델 다운로드 및 `--model` 경로 확인 |
| 카메라 장치가 없음 | USB 카메라 연결과 `/dev/video*` 확인 |
| 카메라를 열 수 없음 | `--device` 번호, 장치 접근 권한, GStreamer 지원 확인 |
| 프레임 수신 실패 또는 `select() timeout` | YUYV 640×480 30fps 지원 여부와 위 GStreamer 테스트 확인 |
| GStreamer 파이프라인 오류 | OpenCV의 GStreamer 지원과 지원 포맷 확인 |
| 영상 창이 나타나지 않음 | 화면 출력이 가능한 데스크톱 환경에서 실행 중인지 확인 |

필터 혼합과 손 크기에 따른 강도 계산은 합성 입력으로 검증했습니다.
현재 YUYV 설정의 실제 카메라 동작은 대상 장치에서 확인해야 합니다.
MediaPipe의 `inference_feedback_manager` 경고는 그 자체만으로 실행 실패를 의미하지 않습니다.

## 코드 설명 순서

1. `open_camera`: 카메라 영상을 가져옵니다.
2. `main`의 손 인식: 좌우 반전한 원본을 RGB로 변환해 MediaPipe에 전달합니다. 손의 관절 좌표 21개를 받습니다.
3. `calculate_angle`: 관절 하나를 중심으로 두 벡터를 만들어 각도를 계산합니다.
4. `count_fingers`: 손가락마다 두 관절의 각도를 확인합니다. 둘 다 기준 이상이면 펴진 손가락으로 셉니다. 엄지는 150도, 나머지는 160도가 기준입니다.
5. `main`의 전환 처리: 같은 개수가 5프레임 연속 나왔을 때 필터를 선택합니다. 주먹도 이 대기 후 원본으로 바뀝니다.
6. `calculate_strength`: 손바닥의 화면 대비 크기를 0~1 강도로 변환합니다. 매 프레임의 측정값을 사용합니다.
7. `make_filter`: 개수에 따라 OpenCV 필터 결과를 만듭니다. 흑백/윤곽선 결과는 BGR로 변환합니다.
8. `apply_filter`: 원본과 필터 결과를 강도에 따라 혼합합니다. 혼합 후 상태 글씨와 강도 막대를 그립니다.

샤프닝은 중심 픽셀에 5를 곱하고 상하좌우 값을 빼 경계를 강조합니다.
색상 변경은 uint8 HSV에서 H값을 40만큼 이동하고 180으로 나눈 나머지를 사용합니다.
덧셈 전 int16으로 변환하여 uint8 덧셈의 넘침을 방지합니다.

## 시연 팁과 한계

- 손바닥이 카메라를 향하도록 하고 한 손 전체가 화면에 나오게 합니다.
- 손가락을 구부리거나 펼치는 자세를 명확하게 보여주세요. 특히 주먹에서는 엄지도 접어야 합니다.
- 각도에 의한 규칙 기반 판정이므로 손의 기울기, 가림, 조명에 따라 개수가 틀릴 수 있습니다.
- 색상 변경은 색이 있는 물체에서, 샤프닝은 무늬나 경계가 있는 장면에서 차이가 잘 보입니다.
- 수업에서 사용한 손 인식 모델을 활용하며, 직접 모델을 학습하지 않습니다.
