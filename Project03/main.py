"""카메라에서 물건을 찾고, v를 누르면 음성으로 질문한다."""

import json
from pathlib import Path
import subprocess


PROJECT_DIR = Path(__file__).resolve().parent


def load_config():
    with open(PROJECT_DIR / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    for key in ("yolo_model", "gemma_model", "whisper_cli", "whisper_model",
                "piper_python", "piper_model"):
        config[key] = str((PROJECT_DIR / config[key]).resolve())
    return config


def detections_to_text(result, width):
    sentences = []
    for box in result.boxes:
        name = result.names[int(box.cls[0].item())]
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        center_x = (x1 + x2) / 2

        # 화면을 세 구역으로 나눠 물건의 위치를 설명한다.
        if center_x < width / 3:
            position = "왼쪽"
        elif center_x < width * 2 / 3:
            position = "가운데"
        else:
            position = "오른쪽"
        sentences.append(f"{name}이 화면 {position}에 있습니다.")

    if not sentences:
        return "탐지된 물건이 없습니다. 실제로 물건이 없다는 뜻은 아닙니다."
    return "\n".join(sentences)


def speech_to_text(config, audio_dir):
    audio_path = audio_dir / "input.wav"
    text_path = audio_dir / "question.txt"
    text_path.unlink(missing_ok=True)
    print(f"지금 질문해 주세요. {config['record_seconds']}초 동안 녹음합니다.", flush=True)
    subprocess.run([
        "arecord", "-q", "-D", config["mic_device"], "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(config["record_seconds"]), str(audio_path),
    ], check=True, timeout=config["record_seconds"] + 10)
    subprocess.run([
        config["whisper_cli"], "-m", config["whisper_model"], "-f", str(audio_path),
        "-l", "ko", "-nt", "-otxt", "-of", str(audio_dir / "question"),
    ], check=True, timeout=120)
    return text_path.read_text(encoding="utf-8").strip()


def ask_gemma(llm, scene_text, question):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": (
                "너는 책상 주변 물건을 설명하는 도우미다. "
                "주어진 탐지 결과만 근거로 한국어 두 문장 이내로 답해라. "
                "위치는 카메라 화면 기준이다. 탐지되지 않은 물건은 확인할 수 없다고 말해라. "
                "색상, 글자, 실제 거리 등 알 수 없는 내용은 추측하지 마라."
            )},
            {"role": "user", "content": f"탐지 결과:\n{scene_text}\n\n질문: {question}"},
        ],
        max_tokens=160,
        temperature=0.2,
    )
    answer = (response["choices"][0]["message"]["content"] or "").strip()
    if not answer:
        raise ValueError("답변을 만들지 못했습니다. 다시 질문해 주세요.")
    return answer


def text_to_speech(text, config, audio_dir):
    output_path = audio_dir / "response.wav"
    subprocess.run([
        config["piper_python"], "-m", "piper", "-m", config["piper_model"],
        "-f", str(output_path), "--", text,
    ], check=True, timeout=120)
    subprocess.run([
        "aplay", "-q", "-D", config["speaker_device"], str(output_path),
    ], check=True, timeout=120)


def main():
    import cv2
    from ultralytics import YOLO
    from llama_cpp import Llama

    config = load_config()
    audio_dir = PROJECT_DIR / "runtime"
    audio_dir.mkdir(exist_ok=True)
    print("모델을 불러오는 중입니다...", flush=True)
    yolo = YOLO(config["yolo_model"])
    llm = Llama(model_path=config["gemma_model"], n_gpu_layers=config["gpu_layers"],
                n_ctx=2048, n_batch=32, n_ubatch=32, verbose=False)
    # 기존 장비에서 사용한 GStreamer 설정으로 카메라를 연다.
    pipeline = (
        f"v4l2src device=/dev/video{config['camera']} ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print("v: 음성 질문 / q: 종료", flush=True)

    try:
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. 장치 번호와 GStreamer 지원을 확인하세요.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            result = yolo.predict(frame, conf=config["confidence"], imgsz=640,
                                  max_det=15, verbose=False)[0]
            cv2.imshow("Desk assistant - v: ask, q: quit", result.plot())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                # 버튼을 누른 장면으로 질문한다. 답변이 끝나면 카메라로 돌아온다.
                scene_text = detections_to_text(result, frame.shape[1])
                print(f"\n탐지 결과:\n{scene_text}", flush=True)
                try:
                    question = speech_to_text(config, audio_dir)
                    if not question:
                        print("음성이 인식되지 않았습니다. 다시 v를 눌러 주세요.")
                        continue
                    print(f"질문: {question}", flush=True)
                    answer = ask_gemma(llm, scene_text, question)
                    print(f"답변: {answer}", flush=True)
                    text_to_speech(answer, config, audio_dir)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    print(f"질문 처리 실패: {error}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")
