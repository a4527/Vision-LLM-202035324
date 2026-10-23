"""손가락 개수로 조작하는 실시간 카메라 필터. 종료: q 또는 ESC."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "src/models/MediaPipe/hand_landmarker.task"
FILTER_NAMES = ("ORIGINAL", "GRAYSCALE", "BLUR", "EDGE", "SHARPEN", "HUE SHIFT")
STABLE_FRAMES = 5
SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
], dtype=np.float32)


def calculate_angle(p1, p2, p3):
    """수업과 동일하게 두 벡터 사이의 각도(도)를 계산한다."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    length = np.linalg.norm(v1) * np.linalg.norm(v2)
    if length == 0:
        return 0.0
    cosine = np.clip(np.dot(v1, v2) / length, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def count_fingers(hand):
    """두 관절이 모두 펴져 있는 손가락을 센다. 좌우 손 공통 규칙."""
    count = 0
    # 각 손가락의 연속된 관절 4개: 엄지, 검지, 중지, 약지, 소지
    for base, joint1, joint2, tip in (
        (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
        (13, 14, 15, 16), (17, 18, 19, 20),
    ):
        # 엄지는 다른 손가락보다 자연스럽게 굽어 있어 기준을 낮춘다.
        threshold = 150 if tip == 4 else 160
        angle1 = calculate_angle(hand[base], hand[joint1], hand[joint2])
        angle2 = calculate_angle(hand[joint1], hand[joint2], hand[tip])
        if angle1 >= threshold and angle2 >= threshold:
            count += 1
    return count


def apply_filter(frame, finger_count):
    """모든 결과를 BGR 3채널로 반환하여 같은 방식으로 글씨를 그린다."""
    if finger_count == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if finger_count == 2:
        return cv2.GaussianBlur(frame, (15, 15), 0)
    if finger_count == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if finger_count == 4:
        return cv2.filter2D(frame, -1, SHARPEN_KERNEL)
    if finger_count == 5:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # uint8 HSV의 H 범위는 0~179. 덧셈 전 int16으로 바꿔 넘침 방지.
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + 40) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame.copy()


def open_camera(camera, device):
    if camera == "webcam":
        # WSL/Linux에서는 V4L2를 직접 사용해 자동 백엔드 선택을 피한다.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(device, backend)
        if cap.isOpened():
            # USB/IP 전달 시 전송량을 줄이기 위해 압축 영상(MJPEG)을 요청한다.
            # 지원 여부에 따라 드라이버가 다른 설정을 선택할 수 있다.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            pixel_format = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
            print(
                f"Camera: device={device}, format={pixel_format!r}, "
                f"size={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, "
                f"fps={cap.get(cv2.CAP_PROP_FPS):g}", flush=True,
            )
        return cap
    if camera == "usb":
        # mp-test.py와 동일한 USB YUYV 카메라 설정.
        source = (
            f"v4l2src device=/dev/video{device} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
        )
    else:
        source = (
            f"nvarguscamerasrc sensor-id={device} ! "
            "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
        )
    pipeline = source + (
        "queue leaky=downstream max-size-buffers=1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=("usb", "csi", "webcam"), default="usb")
    parser.add_argument("--device", type=int, default=0, help="카메라 번호 (기본: 0)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="손 인식 모델 경로")
    args = parser.parse_args()

    # 필터 함수만 따로 확인할 때는 MediaPipe가 필요하지 않다.
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        parser.error("MediaPipe가 없습니다. Project01/README.md의 설치 안내를 확인하세요.")
    if not args.model.is_file():
        parser.error(f"손 인식 모델이 없습니다: {args.model}\nREADME.md의 다운로드 안내를 확인하세요.")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        num_hands=1,
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        cap = open_camera(args.camera, args.device)
        try:
            if not cap.isOpened():
                raise RuntimeError("카메라를 열 수 없습니다. 카메라 종류와 --device 번호를 확인하세요.")

            active_count = 0
            previous_count = None
            consecutive_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        "카메라는 열렸지만 영상 프레임을 받지 못했습니다. "
                        "위 Camera 설정과 경고 메시지를 확인하세요. "
                        "WSL에서는 usbipd 연결 및 카메라 전송 형식 확인이 필요합니다."
                    )

                # 거울처럼 좌우 반전. 필터 적용 전 원본에서 손을 인식한다.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                detected_count = count_fingers(hand) if hand is not None else None

                if detected_count is None:
                    # 손이 없으면 즉시 원본으로 복귀하고 전환 대기를 초기화.
                    active_count = 0
                    previous_count = None
                    consecutive_frames = 0
                else:
                    if detected_count == previous_count:
                        consecutive_frames += 1
                    else:
                        previous_count = detected_count
                        consecutive_frames = 1
                    if consecutive_frames >= STABLE_FRAMES:
                        active_count = detected_count

                output = apply_filter(frame, active_count)
                # 필터 적용 후 글씨를 그려서 어떤 필터에서도 읽을 수 있게 한다.
                detected_text = str(detected_count) if hand is not None else "NO HAND"
                lines = (
                    f"Fingers: {detected_text}",
                    f"Filter: {active_count} - {FILTER_NAMES[active_count]}",
                    "1:Gray  2:Blur  3:Edge  4:Sharp  5:Hue",
                    "0 / No hand: Original    Q / ESC: Quit",
                )
                for i, text in enumerate(lines):
                    position = (10, 25 + i * 25)
                    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("Finger Camera Filters", output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Finger Camera Filters", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
