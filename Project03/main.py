"""찾아줘: 음성으로 물건을 등록하면 화면에 나타났을 때 한 번 알려 준다."""

import json
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
import wave


PROJECT_DIR = Path(__file__).resolve().parent


def memory_summary():
    # Jetson에서 모델을 올린 전후의 가용 RAM과 전체 SWAP 사용량을 확인한다.
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0])
        available = values["MemAvailable"] // 1024
        swap_used = (values["SwapTotal"] - values["SwapFree"]) // 1024
        return f"가용 RAM={available} MiB, 전체 SWAP 사용={swap_used} MiB"
    except (OSError, KeyError, ValueError):
        return "메모리 정보를 읽지 못했습니다."


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
        max_rms = 0.0
        while True:
            data = audio.readframes(sample_rate // 50)
            if not data:
                break
            samples = array("h", data)
            if sys.byteorder != "little":
                samples.byteswap()
            average = sum(samples) / len(samples)
            rms = (sum((value - average) ** 2 for value in samples) / len(samples)) ** 0.5
            max_rms = max(max_rms, rms)
            if rms >= threshold:
                active_samples += len(samples)
        active_seconds = active_samples / sample_rate
        print(f"녹음 확인: 최대 RMS={max_rms:.0f}, 기준={threshold}, "
              f"기준 이상={active_seconds:.2f}초 (필요: 0.30초)", flush=True)
        return active_seconds >= 0.3


def speech_to_text(config, audio_dir):
    audio_path = audio_dir / "input.wav"
    text_path = audio_dir / "question.txt"
    text_path.unlink(missing_ok=True)
    print(f"녹음 장치={config['mic_device']}, 무음 기준={config['silence_threshold']}", flush=True)
    print(f"설정 파일: {PROJECT_DIR / 'config.json'}", flush=True)
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


def load_gemma(config):
    from llama_cpp import Llama

    print(f"Gemma를 처음 한 번 불러옵니다 (컨텍스트 {config['context_window']}토큰)...", flush=True)
    return Llama(model_path=config["gemma_model"], n_gpu_layers=config["gpu_layers"],
                 n_ctx=config["context_window"], n_batch=16, n_ubatch=16, n_threads=2,
                 n_threads_batch=2, verbose=False)


def choose_target(llm, request, class_names):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": (
                "Parse a Korean object-finding request as JSON. "
                f"object must be one of: {','.join(class_names)}. "
                f"color must be one of: {','.join(COLOR_NAMES)}; no color means any. "
                "name is the short Korean object noun. "
                'Example: 초록사과 -> {"object":"apple","color":"green","name":"사과"}. '
                "For unsupported objects/colors, multiple targets or extra conditions, "
                'return {"object":"unknown","color":"any","name":""}. Do not substitute.'
            )},
            {"role": "user", "content": request},
        ],
        response_format={"type": "json_object"},
        max_tokens=96, temperature=0.0,
    )
    # 실제 입력량에는 사용자 말 외에 지시문, 클래스 목록, 채팅 형식도 포함된다.
    prompt_tokens = response.get("usage", {}).get("prompt_tokens")
    if prompt_tokens is not None:
        print(f"Gemma 전체 입력: {prompt_tokens}토큰", flush=True)
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


def register_request(llm, config, class_names, audio_dir):
    request = speech_to_text(config, audio_dir)
    if not request:
        print("요청이 없어 기존 상태를 유지합니다.", flush=True)
        return None
    print(f"요청: {request}", flush=True)
    print("Gemma가 찾을 조건을 해석하는 중입니다...", flush=True)
    return choose_target(llm, request, class_names)


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
    print(f"모델 로딩 전: {memory_summary()}", flush=True)
    yolo = YOLO(config["yolo_model"])
    llm = load_gemma(config)
    print(f"모델 로딩 후: {memory_summary()}", flush=True)
    target = None
    found_count = 0
    last_check = 0
    class_ids = {name: number for number, name in yolo.names.items()}
    print(f"이 YOLO 모델이 지원하는 물건 종류: {len(class_ids)}개", flush=True)
    support_text = "지원 물체\n" + "\n".join(
        f"{number}: {name}" for name, number in class_ids.items()
    ) + "\n\n지원 색상: " + ", ".join(name for name in COLOR_NAMES.values() if name)
    support_path = audio_dir / "supported_objects.txt"
    support_path.write_text(support_text + "\n", encoding="utf-8")
    print(f"지원 목록: {support_path} (h 키로 터미널에 출력)", flush=True)
    # 오래 걸리는 음성 처리만 작업 스레드 하나에서 실행한다.
    worker = ThreadPoolExecutor(max_workers=1)
    pending = None
    pending_kind = None
    ignore_request = False
    cap = None
    failures = 0
    print("준비 완료. v: 음성 등록 / c: 취소 / h: 지원 목록 / q: 종료", flush=True)

    try:
        cap = open_camera(cv2, config)
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. 장치 번호와 GStreamer 지원을 확인하세요.")
        while True:
            if pending is not None and pending.done():
                finished = pending
                kind = pending_kind
                pending = None
                try:
                    new_target = finished.result()
                    if kind == "request" and not ignore_request and new_target:
                        target = new_target
                        found_count = 0
                        last_check = 0
                        message = f"{target['label']}이 보이면 알려드릴게요."
                        print(message, flush=True)
                        pending = worker.submit(text_to_speech, message, config, audio_dir)
                        pending_kind = "speech"
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    print(f"음성 처리 실패: {error}", flush=True)

            read_started = time.perf_counter()
            ok, frame = cap.read()
            read_elapsed = time.perf_counter() - read_started
            if read_elapsed >= 1.0:
                print(f"카메라 수신 지연: {read_elapsed:.1f}초 / {memory_summary()}", flush=True)
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
            if target and pending is None and time.monotonic() - last_check >= config["check_interval"]:
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
                    pending = worker.submit(text_to_speech, message, config, audio_dir)
                    pending_kind = "speech"

            if pending is not None:
                status = "Listening / processing..." if pending_kind == "request" else "Speaking..."
            else:
                status = f"{target['color']} {target['object']}" if target else "v: new request"
            cv2.putText(output, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
            display_started = time.perf_counter()
            cv2.imshow("Find it - v: request, c: cancel, q: quit", output)
            key = cv2.waitKey(1) & 0xFF
            display_elapsed = time.perf_counter() - display_started
            if display_elapsed >= 1.0:
                print(f"화면 갱신 지연: {display_elapsed:.1f}초 / {memory_summary()}", flush=True)
            if key == ord("q"):
                break
            if key == ord("h"):
                print(support_text, flush=True)
            if key == ord("c"):
                ignore_request = True
                target = None
                found_count = 0
                print("찾기를 취소했습니다.", flush=True)
            if key == ord("v") and pending is None:
                ignore_request = False
                found_count = 0
                pending_kind = "request"
                pending = worker.submit(register_request, llm, config, list(class_ids), audio_dir)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # 진행 중인 추론이 끝난 뒤 모델을 닫아 메모리 접근 충돌을 막는다.
        if pending is not None and not pending.done():
            print("진행 중인 음성 처리를 마친 뒤 종료합니다.", flush=True)
        worker.shutdown(wait=True)
        llm.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
