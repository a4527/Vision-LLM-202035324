"""찾아줘: 음성으로 물건을 등록하면 화면에 나타났을 때 한 번 알려 준다."""

import json
from array import array
from pathlib import Path
import subprocess
import sys
import time
import wave


PROJECT_DIR = Path(__file__).resolve().parent


def load_config():
    with open(PROJECT_DIR / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    for key in ("yolo_model", "gemma_model", "whisper_cli", "whisper_model",
                "piper_python", "piper_model"):
        config[key] = str((PROJECT_DIR / config[key]).resolve())
    return config


COLOR_NAMES = {
    "any": "", "red": "빨간", "orange": "주황색", "yellow": "노란",
    "green": "초록색", "blue": "파란", "purple": "보라색",
    "white": "흰", "black": "검은",
}

# OpenCV HSV 범위. 빨간색은 색상환 양 끝에 있어 두 구간을 확인한다.
COLOR_RANGES = {
    "red": [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    "orange": [((11, 80, 60), (24, 255, 255))],
    "yellow": [((25, 80, 60), (34, 255, 255))],
    "green": [((35, 80, 60), (85, 255, 255))],
    "blue": [((86, 80, 60), (130, 255, 255))],
    "purple": [((131, 80, 60), (169, 255, 255))],
    "white": [((0, 0, 180), (179, 50, 255))],
    "black": [((0, 0, 0), (179, 255, 50))],
}


def matches_color(image, color, min_ratio):
    import cv2

    if image.size == 0:
        return False
    if color == "any":
        return True
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *COLOR_RANGES[color][0])
    for lower, upper in COLOR_RANGES[color][1:]:
        mask |= cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) / mask.size >= min_ratio


def find_position(result, target, frame, min_ratio):
    height, width = frame.shape[:2]
    for box in result.boxes:
        if result.names[int(box.cls[0].item())] != target["object"]:
            continue
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().tolist())
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # 박스 바깥쪽 배경의 영향을 줄이려고 가운데 60% 영역을 본다.
        margin_x, margin_y = (x2 - x1) // 5, (y2 - y1) // 5
        image = frame[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]
        if not matches_color(image, target["color"], min_ratio):
            continue
        center_x = (x1 + x2) / 2
        if center_x < width / 3:
            return "왼쪽"
        if center_x < width * 2 / 3:
            return "가운데"
        return "오른쪽"
    return None


def has_voice(audio_path, threshold):
    # 20ms씩 소리 크기를 확인한다. 짧은 클릭음만으로 질문을 시작하지 않는다.
    with wave.open(str(audio_path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise ValueError("녹음 형식은 16비트 모노여야 합니다.")
        sample_rate = audio.getframerate()
        active_samples = 0
        while True:
            data = audio.readframes(sample_rate // 50)
            if not data:
                break
            samples = array("h", data)
            if sys.byteorder != "little":
                samples.byteswap()
            average = sum(samples) / len(samples)
            rms = (sum((value - average) ** 2 for value in samples) / len(samples)) ** 0.5
            if rms >= threshold:
                active_samples += len(samples)
        return active_samples >= sample_rate * 0.3


def speech_to_text(config, audio_dir):
    audio_path = audio_dir / "input.wav"
    text_path = audio_dir / "question.txt"
    text_path.unlink(missing_ok=True)
    print(f"찾을 물건을 말해 주세요. {config['record_seconds']}초 동안 녹음합니다.", flush=True)
    subprocess.run([
        "arecord", "-q", "-D", config["mic_device"], "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(config["record_seconds"]), str(audio_path),
    ], check=True, timeout=config["record_seconds"] + 10)
    if not has_voice(audio_path, config["silence_threshold"]):
        print("소리가 작거나 무음입니다. 답변하지 않고 미리보기로 돌아갑니다.", flush=True)
        return ""
    subprocess.run([
        config["whisper_cli"], "-m", config["whisper_model"], "-f", str(audio_path),
        "-l", "ko", "-nt", "-otxt", "-of", str(audio_dir / "question"),
    ], check=True, timeout=120)
    question = text_path.read_text(encoding="utf-8").strip()
    # Whisper가 문장 대신 무음/음악 표시만 출력한 경우도 건너뛴다.
    if question.startswith(("[", "(")) and question.endswith(("]", ")")):
        return ""
    if not any(letter.isalnum() for letter in question):
        return ""
    return question


def choose_target(request, config, class_names):
    from llama_cpp import Llama

    print("Gemma가 찾을 물건을 정하는 중입니다...", flush=True)
    llm = Llama(model_path=config["gemma_model"], n_gpu_layers=config["gpu_layers"],
                n_ctx=1024, n_batch=16, n_ubatch=16, n_threads=2,
                n_threads_batch=2, verbose=False)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": (
                    "물건 찾기 요청을 JSON 하나로 바꿔라. "
                    f"object는 다음 YOLO 클래스 중 하나: {', '.join(class_names)}. "
                    f"color는 다음 중 하나: {', '.join(COLOR_NAMES)}. "
                    "색을 말하지 않으면 any다. name은 물건 종류의 짧은 한국어 이름이다. "
                    '예: 빨간 컵이 보이면 알려줘 -> {"object":"cup","color":"red","name":"컵"}. '
                    "대상이 여러 개, 미지원 물건/색상, 색 외 추가 조건은 object를 unknown으로 해라. "
                    "임의의 비슷한 물건이나 색으로 바꾸지 마라."
                )},
                {"role": "user", "content": request},
            ],
            response_format={"type": "json_object"},
            max_tokens=96, temperature=0.0,
        )
        target = json.loads(response["choices"][0]["message"]["content"] or "{}")
        if not isinstance(target, dict):
            raise ValueError("요청을 해석하지 못했습니다. 물건 하나를 다시 말해 주세요.")
        name = target.get("name")
        if (not isinstance(target.get("object"), str) or target["object"] not in class_names
                or not isinstance(target.get("color"), str) or target["color"] not in COLOR_NAMES
                or not isinstance(name, str) or not name.strip() or len(name) > 30):
            raise ValueError("지원하는 물건과 색상 하나로 다시 말해 주세요.")
        target["label"] = f"{COLOR_NAMES[target['color']]} {name.strip()}".strip()
        return target
    finally:
        # 관찰할 때는 Gemma가 필요 없으므로 모델 메모리를 해제한다.
        llm.close()


def text_to_speech(text, config, audio_dir):
    output_path = audio_dir / "response.wav"
    subprocess.run([
        config["piper_python"], "-m", "piper", "-m", config["piper_model"],
        "-f", str(output_path), "--", text,
    ], check=True, timeout=120)
    subprocess.run([
        "aplay", "-q", "-D", config["speaker_device"], str(output_path),
    ], check=True, timeout=120)


def open_camera(cv2, config):
    # 장비 입력은 기존 형식을 유지하고, 처리할 영상만 작게 줄인다.
    pipeline = (
        f"v4l2src device=/dev/video{config['camera']} ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "videorate drop-only=true ! video/x-raw,framerate=10/1 ! "
        "videoscale ! video/x-raw,width=320,height=240 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def main():
    import cv2
    import torch
    from ultralytics import YOLO

    # CPU를 전부 사용하지 않도록 스레드 수를 제한한다.
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config = load_config()
    audio_dir = PROJECT_DIR / "runtime"
    audio_dir.mkdir(exist_ok=True)
    print("YOLO를 불러오는 중입니다...", flush=True)
    yolo = YOLO(config["yolo_model"])
    target = None
    found_count = 0
    last_check = 0
    class_ids = {name: number for number, name in yolo.names.items()}
    print(f"이 YOLO 모델이 지원하는 물건 종류: {len(class_ids)}개", flush=True)
    cap = open_camera(cv2, config)
    failures = 0
    print("v: 찾을 물건 등록 / c: 찾기 취소 / q: 종료", flush=True)

    try:
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. 장치 번호와 GStreamer 지원을 확인하세요.")
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                found_count = 0
                print(f"카메라 읽기 실패 ({failures}/3)", flush=True)
                if failures >= 3:
                    print("장치 연결, 지원 영상 형식, 전원 상태를 확인해 주세요.")
                    break
                cap.release()
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
                time.sleep(0.5)
                cap = open_camera(cv2, config)
                continue
            failures = 0

            output = frame
            if target and time.monotonic() - last_check >= config["check_interval"]:
                result = yolo.predict(
                    frame, conf=config["confidence"], imgsz=320,
                    classes=[class_ids[target["object"]]], max_det=5, verbose=False,
                )[0]
                last_check = time.monotonic()
                position = find_position(result, target, frame, config["color_min_ratio"])
                found_count = found_count + 1 if position else 0
                output = result.plot()
                if found_count >= config["confirm_count"]:
                    message = f"찾으시던 {target['label']}이 화면 {position}에 보입니다."
                    print(message, flush=True)
                    # 한 번 알린 뒤에는 찾기를 끝내 같은 물건을 계속 읽지 않는다.
                    target = None
                    found_count = 0
                    cap.release()
                    try:
                        text_to_speech(message, config, audio_dir)
                    except (OSError, subprocess.SubprocessError) as error:
                        print(f"음성 재생 실패: {error}", flush=True)
                    finally:
                        cap = open_camera(cv2, config)

            status = f"{target['color']} {target['object']}" if target else "v: new request"
            cv2.putText(output, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
            cv2.imshow("Find it - v: request, c: cancel, q: quit", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                target = None
                found_count = 0
                print("찾기를 취소했습니다.", flush=True)
            if key == ord("v"):
                cap.release()
                try:
                    request = speech_to_text(config, audio_dir)
                    if not request:
                        print("요청이 없어 기존 상태로 돌아갑니다.", flush=True)
                        continue
                    print(f"요청: {request}", flush=True)
                    new_target = choose_target(request, config, list(class_ids))
                    target = new_target
                    found_count = 0
                    last_check = 0
                    print(f"{target['label']}을 찾습니다. 발견하면 한 번 알려드립니다.", flush=True)
                    text_to_speech(f"{target['label']}이 보이면 알려드릴게요.", config, audio_dir)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    print(f"요청 처리 실패: {error}", flush=True)
                finally:
                    cap = open_camera(cv2, config)
                    # 처리 중 쌓인 키 입력이 다음 요청을 다시 시작하지 않게 비운다.
                    for _ in range(100):
                        if cv2.waitKey(1) == -1:
                            break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
