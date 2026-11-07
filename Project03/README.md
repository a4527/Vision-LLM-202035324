# Project03 - 책상 주변 음성 도우미

카메라에 보이는 물건을 YOLO로 탐지하고, 음성으로 질문하면 Gemma가 답변을 만든 뒤 Piper가 읽어 주는 프로그램이다. 예를 들어 컵을 화면 왼쪽에 놓고 “컵이 어느 쪽에 있어?”라고 질문할 수 있다.

## 처리 순서

```text
카메라 → YOLO → 물건 종류와 위치를 문장으로 정리
                        ↓ v 키 입력
마이크 → Whisper → 질문 + 탐지 결과 → Gemma → Piper → 스피커
```

`main.py` 하나에서 위 순서대로 실행한다. 질문 처리 중에는 카메라 화면이 잠시 멈추고, 음성 재생이 끝나면 다시 탐지를 시작한다. 질문의 기준은 `v`를 눌렀을 때의 장면이다.

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

카메라는 기존 `yolotest.py`와 같은 GStreamer pipeline으로 연결한다. `/dev/video0`의 YUY2 영상을 640×480, 30 FPS로 받아 OpenCV가 사용할 BGR 형식으로 바꾼다. 프레임이 밀리지 않도록 오래된 프레임은 버리고 버퍼에는 한 장만 남기도록 설정했다.

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
| `gpu_layers` | Gemma의 GPU 배치 레이어 수, 기본 -1 |

장치 번호는 `arecord -l`, `aplay -l`로 확인한다. 예를 들어 마이크가 card 3, device 0이면 `mic_device`를 `plughw:3,0`으로 바꾼다.

## 실행

Jetson 데스크톱의 터미널에서 수업용 Python 환경을 활성화한 뒤 실행한다.

```bash
cd Project03
python3 main.py
```

1. 모델 로딩이 끝나면 탐지 화면이 표시된다.
2. 카메라 창을 선택하고 `v`를 누른다.
3. 터미널에 녹음 안내가 나오면 5초 안에 질문한다.
4. 터미널에서 인식한 질문과 답변을 확인하고 음성 답변을 듣는다.
5. 화면이 다시 움직이면 다음 질문을 할 수 있다. 종료는 `q` 또는 `Ctrl+C`다.

카메라가 열리지 않으면 `camera` 번호와 OpenCV의 GStreamer 지원 여부를 확인한다. 녹음이나 재생 오류가 나면 ALSA 장치 번호를 확인한다. PulseAudio가 장치를 점유한 경우에는 수업에서처럼 `pasuspender -- python3 main.py`로 실행해 본다. 메모리가 부족하면 `gpu_layers` 값을 줄여 일부 연산을 CPU에서 처리할 수 있다.

## 코드 설명

| 함수 | 하는 일 |
|---|---|
| `load_config()` | 모델 경로와 장치 설정을 읽는다. |
| `detections_to_text()` | 탐지한 물건의 이름과 화면상 위치를 문장으로 바꾼다. |
| `speech_to_text()` | 녹음한 뒤 Whisper가 만든 질문 텍스트를 읽는다. |
| `ask_gemma()` | 탐지 결과와 질문을 Gemma에 전달한다. |
| `text_to_speech()` | Piper로 답변 음성을 만들고 재생한다. |
| `main()` | 카메라를 반복해서 읽고 키 입력에 따라 위 함수를 호출한다. |

위치는 박스 중심의 가로 좌표로 구한다. 중심이 화면 너비의 1/3보다 작으면 왼쪽, 2/3보다 작으면 가운데, 나머지는 오른쪽이다. 신뢰도 0.4 이상인 객체를 최대 15개 탐지한다.

Gemma에는 “탐지 결과만 근거로 한국어 두 문장 이내로 답하라”는 지시를 준다. 객체가 탐지되지 않았다고 해서 실제로 없다고 단정하지 않도록 안내한다. 음성 처리 오류가 발생하면 내용을 출력하고 카메라 반복문으로 돌아간다.

## 발표할 때 설명할 핵심

“YOLO의 결과를 Gemma가 읽을 수 있는 문장으로 바꿨습니다. 사용자가 v를 누르면 Whisper로 질문을 받아 탐지 결과와 함께 Gemma에 넣습니다. 마지막으로 Piper가 답변을 읽어 줍니다. 한 파일에서 순서대로 실행해서 처리 흐름을 쉽게 따라갈 수 있게 했습니다.”

시연에서는 컵의 위치 질문, 보이는 물건의 종류 질문, 탐지되지 않은 물건 질문 순서로 진행한다.

이미지를 Gemma에 직접 전달하는 방식은 아니므로 색상, 글자, 실제 거리는 알 수 없다. YOLO의 탐지 오류와 Gemma의 잘못된 답변도 발생할 수 있다. 질문 중 화면이 멈추는 점이 이 구현의 한계다.

장비 시연은 별도로 진행한다. 실제 응답 시간과 성능 측정값은 아직 작성하지 않았다. 코드와 문서 정리에 생성형 AI의 도움을 사용했다.

상위 README의 제출 방식에 맞춰 실행 파일, 의존성 안내, 발표 PDF를 `Project03`에 모았다. 장비 확인 후 GitHub에 제출할 때는 `Project03: 책상 주변 음성 도우미 구현`처럼 구체적인 커밋 메시지를 사용한다.
