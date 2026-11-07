# Project03 - 책상 주변 음성 도우미

평소에는 카메라 미리보기를 띄우고, `v`를 누르면 YOLO로 물건을 탐지한다. 음성으로 질문하면 Gemma가 답변을 만든 뒤 Piper가 읽어 주는 프로그램이다. 예를 들어 컵을 화면 왼쪽에 놓고 “컵이 어느 쪽에 있어?”라고 질문할 수 있다.

## 처리 순서

```text
카메라 미리보기 → v 키 입력 → YOLO → 물건 종류와 위치를 문장으로 정리
마이크 → Whisper → 질문 + 탐지 결과 → Gemma → Piper → 스피커
```

`main.py` 하나에서 위 순서대로 실행한다. 질문 중에는 카메라를 닫아 부하를 줄이고, 음성 재생이 끝나면 다시 미리보기를 시작한다. 탐지 박스는 `v`를 눌렀을 때 표시한다. 질문의 기준은 `v`를 눌렀을 때의 장면이다.

## 제출 파일

- `main.py`: 카메라 탐지와 음성 대화 전체 코드
- `config.json`: 모델 경로와 장치 설정
- `requirements.txt`: Python 의존성
- `presentation.md`: 발표 내용 원본
- `output/pdf/presentation.pdf`: 제출용 발표 자료

실행 중 생성되는 녹음, 질문 텍스트, 응답 음성은 `runtime/`에 저장되고 다음 질문에서 덮어쓴다. 이 폴더는 Git에서 제외했다.

## 준비와 설정

수업에서 사용한 Jetson의 Python/PyTorch/CUDA 환경을 사용한다. USB 카메라, 마이크와 스피커를 연결한다.

필요한 항목은 `ultralytics`, `llama-cpp-python`, OpenCV(`cv2`), Whisper.cpp, Piper와 각 모델 파일이다. 녹음과 재생에는 `alsa-utils`의 `arecord`, `aplay`를 사용한다. Piper는 수업에서 만든 별도 가상환경에 설치한다.

```bash
cd Project03
python3 -m pip install -r requirements.txt
```

위 명령은 Python 패키지 설치용이며 Jetson의 CUDA 빌드 설정까지 구성하지는 않는다. 수업에서 설치한 CUDA 지원 환경과 GStreamer 지원 OpenCV를 사용한다.

카메라는 기존 `yolotest.py`의 입력 형식인 `/dev/video0`, YUY2, 640×480, 30 FPS를 유지한다. pipeline 중간에서 10 FPS와 320×240으로 줄인 뒤 BGR로 바꿔 OpenCV에 전달한다. 카메라 자체의 USB 전송량까지 줄인 것은 아니다. 오래된 프레임은 버리고 버퍼에는 한 장만 남긴다.

`config.json`에서 다음 항목을 확인한다. 상대 경로는 `Project03` 기준이며 절대 경로로 바꿔도 된다. 큰 모델 파일과 Whisper/Piper 실행 도구는 별도로 준비한다.

| 설정 | 설명 |
|---|---|
| `yolo_model` | 기본값은 저장소의 `../src/models/YOLO/yolo11n.pt` |
| `gemma_model` | 수업에서 사용한 Gemma GGUF 파일 경로 |
| `whisper_cli`, `whisper_model` | Whisper 실행 파일과 `ggml-base.bin` 경로 |
| `piper_python` | Piper 가상환경의 Python 실행 파일 |
| `piper_model` | 한국어 `.onnx` 모델 경로. 옆에 `.onnx.json`도 필요 |
| `camera` | 카메라 장치 번호. 기본 0이면 pipeline에서 `/dev/video0` 사용 |
| `mic_device`, `speaker_device` | ALSA 장치 이름, 기본 `default` |
| `record_seconds` | 녹음 시간, 기본 5초 |
| `confidence` | YOLO 탐지 신뢰도 기준, 기본 0.4 |
| `gpu_layers` | Gemma의 GPU 배치 레이어 수. Nano용 기본값 0은 CPU 실행 |

장치 번호는 `arecord -l`, `aplay -l`로 확인한다. 예를 들어 마이크가 card 3, device 0이면 `mic_device`를 `plughw:3,0`으로 바꾼다.

## 실행

Jetson 데스크톱의 터미널에서 수업용 Python 환경을 활성화한 뒤 실행한다.

```bash
cd Project03
python3 main.py
```

1. YOLO 로딩이 끝나면 카메라 미리보기가 표시된다.
2. 카메라 창을 선택하고 `v`를 누른다.
3. 탐지 박스가 표시되고 녹음 안내가 나오면 5초 안에 질문한다. Gemma는 첫 질문에서 불러오므로 첫 응답은 더 오래 걸린다.
4. 터미널에서 인식한 질문과 답변을 확인하고 음성 답변을 듣는다.
5. 화면이 다시 움직이면 다음 질문을 할 수 있다. 종료는 `q` 또는 `Ctrl+C`다.

카메라가 열리지 않으면 `camera` 번호와 OpenCV의 GStreamer 지원 여부를 확인한다. 녹음이나 재생 오류가 나면 ALSA 장치 번호를 확인한다. PulseAudio가 장치를 점유한 경우에는 수업에서처럼 `pasuspender -- python3 main.py`로 실행해 본다. Nano용 설정은 Gemma를 CPU에서 실행하므로 답변이 느릴 수 있다. GPU와 CPU를 바꾸는 것만으로 전체 RAM 부족이 해결되지는 않는다.

## 코드 설명

| 함수 | 하는 일 |
|---|---|
| `load_config()` | 모델 경로와 장치 설정을 읽는다. |
| `detections_to_text()` | 탐지한 물건의 이름과 화면상 위치를 문장으로 바꾼다. |
| `speech_to_text()` | 녹음한 뒤 Whisper가 만든 질문 텍스트를 읽는다. |
| `ask_gemma()` | 탐지 결과와 질문을 Gemma에 전달한다. |
| `text_to_speech()` | Piper로 답변 음성을 만들고 재생한다. |
| `open_camera()` | 기존 입력을 받아 320×240, 10 FPS의 카메라 영상을 연다. |
| `main()` | 미리보기를 표시하고 v 입력 시 탐지와 음성 대화를 순서대로 호출한다. |

위치는 박스 중심의 가로 좌표로 구한다. 중심이 화면 너비의 1/3보다 작으면 왼쪽, 2/3보다 작으면 가운데, 나머지는 오른쪽이다. 신뢰도 0.4 이상인 객체를 최대 5개 탐지한다. YOLO 입력 크기는 640에서 320으로 줄였다. 작은 물건의 탐지 정확도가 낮아질 수 있다.

Gemma에는 “탐지 결과만 근거로 한국어 두 문장 이내로 답하라”는 지시를 준다. 객체가 탐지되지 않았다고 해서 실제로 없다고 단정하지 않도록 안내한다. 음성 처리 오류가 발생하면 내용을 출력하고 카메라 반복문으로 돌아간다.

## Nano에서 낮춘 부분

- 상시 YOLO 추론을 없애고 질문할 때 한 번만 탐지한다.
- 프로그램에 전달할 영상은 320×240, 10 FPS로 줄인다.
- 질문 중에는 카메라를 닫고, 끝나면 다시 연다.
- Gemma 컨텍스트는 2048에서 1024, 배치는 32에서 16, 최대 출력은 160에서 96토큰으로 줄인다.
- Gemma는 첫 질문에 불러오고 재사용한다. CPU 스레드는 2개로 제한한다.
- 카메라 읽기 실패 시 잠시 기다렸다 다시 연결한다. 연속 3회 실패하면 종료한다.

**이 설정으로 Nano 실행이 보장되지는 않는다.** 컨텍스트를 줄여도 Gemma 모델 자체의 크기는 그대로다. 첫 질문의 모델 로딩에서 종료되거나 `Killed`가 나오면 RAM 사용량과 운영체제 로그를 확인하고, 현재 실행 도구와 호환되는 더 작은 GGUF 모델을 준비해야 한다. 실제 Nano에서의 패키지와 모델 호환성도 확인해야 한다.

카메라 읽기 오류는 사양 부족으로 단정할 수 없다. 모델을 불러오기 전 미리보기부터 실패한다면 장치 번호, USB 연결, 카메라 점유와 지원 영상 형식도 확인한다. 장비에서 다음 명령으로 상태를 확인할 수 있다.

```bash
free -h
tegrastats
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

`tegrastats`는 별도 터미널에서 확인하고 Ctrl+C로 종료한다. `v4l2-ctl`이 없다면 장비의 v4l-utils 설치 여부를 확인한다. 카메라의 출력 크기와 FPS까지 낮추려면 지원 목록에 있는 조합을 사용해야 한다.

## 발표할 때 설명할 핵심

“YOLO의 결과를 Gemma가 읽을 수 있는 문장으로 바꿨습니다. 사용자가 v를 누르면 YOLO 탐지를 한 번 실행하고 Whisper로 질문을 받아 탐지 결과와 함께 Gemma에 넣습니다. 마지막으로 Piper가 답변을 읽어 줍니다. 한 파일에서 순서대로 실행해서 처리 흐름을 쉽게 따라갈 수 있게 했습니다.”

시연에서는 컵의 위치 질문, 보이는 물건의 종류 질문, 탐지되지 않은 물건 질문 순서로 진행한다.

이미지를 Gemma에 직접 전달하는 방식은 아니므로 색상, 글자, 실제 거리는 알 수 없다. YOLO의 탐지 오류와 Gemma의 잘못된 답변도 발생할 수 있다. 질문 중 화면이 멈추는 점이 이 구현의 한계다.

장비 시연은 별도로 진행한다. 실제 응답 시간과 성능 측정값은 아직 작성하지 않았다. 코드와 문서 정리에 생성형 AI의 도움을 사용했다.

상위 README의 제출 방식에 맞춰 실행 파일, 의존성 안내, 발표 PDF를 `Project03`에 모았다. 장비 확인 후 GitHub에 제출할 때는 `Project03: 책상 주변 음성 도우미 구현`처럼 구체적인 커밋 메시지를 사용한다.
