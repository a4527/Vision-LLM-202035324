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


def calculate_strength(hand, frame_shape, far_size=0.08, near_size=0.30):
    """손목(0)~중지 시작 관절(9) 길이를 화면 높이로 나눠 강도로 변환."""
    height, width = frame_shape[:2]
    dx = (hand[9].x - hand[0].x) * width
    dy = (hand[9].y - hand[0].y) * height
    # x, y의 정규화 기준이 다르므로 픽셀 거리로 계산한 뒤 높이로 나눈다.
    palm_size = np.hypot(dx, dy) / height
    return float(np.clip((palm_size - far_size) / (near_size - far_size), 0, 1))


def make_filter(frame, finger_count):
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


def apply_filter(frame, finger_count, strength=1.0):
    """원본과 필터 결과를 혼합한다. 0=원본, 1=필터 최대 효과."""
    strength = float(np.clip(strength, 0, 1))
    if finger_count == 0 or strength == 0:
        return frame.copy()
    filtered = make_filter(frame, finger_count)
    return cv2.addWeighted(frame, 1 - strength, filtered, strength, 0)


def open_camera(device):
    """02 노트북의 YUYV 파이프라인. GStreamer에서는 YUY2로 지정한다."""
    pipeline = (
        f"v4l2src device=/dev/video{device} ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "queue leaky=downstream max-size-buffers=1 "
        "max-size-bytes=0 max-size-time=0 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="카메라 번호 (기본: 0)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="손 인식 모델 경로")
    parser.add_argument("--debug", action="store_true", help="실행 단계와 첫 프레임 처리 상태 출력")
    parser.add_argument("--far-size", type=float, default=0.08,
                        help="강도 0%%가 되는 손바닥 길이/화면 높이 비율 (기본: 0.08)")
    parser.add_argument("--near-size", type=float, default=0.30,
                        help="강도 100%%가 되는 손바닥 길이/화면 높이 비율 (기본: 0.30)")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device는 0 이상의 카메라 번호여야 합니다.")
    if not 0 <= args.far_size < args.near_size <= 1:
        parser.error("0 <= --far-size < --near-size <= 1이어야 합니다.")

    def log(message):
        if args.debug:
            print(f"[DEBUG] {message}", flush=True)

    log(f"Python: {sys.executable}; OpenCV: {cv2.__version__} ({cv2.__file__})")
    if not args.model.is_file():
        parser.error(f"손 인식 모델이 없습니다: {args.model}\nREADME.md의 다운로드 안내를 확인하세요.")

    # exK.py처럼 카메라를 먼저 시작한다. 손 인식 모델 초기화와 분리해 확인한다.
    log("카메라 열기 시작 (YUYV 640x480 30fps)")
    cap = open_camera(args.device)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"/dev/video{args.device} 카메라를 열 수 없습니다. "
                "장치 번호, 접근 권한, OpenCV의 GStreamer 지원 및 "
                "YUYV 640x480 30fps 지원 여부를 확인하세요."
            )
        log("카메라 열기 완료; 첫 프레임 읽기 시작")
        ret, preview = cap.read()
        if not ret:
            raise RuntimeError("모델 초기화 전 카메라의 첫 프레임을 받지 못했습니다.")
        log(f"첫 프레임 수신 완료: {preview.shape}; MediaPipe import 시작")

        # 카메라 입력을 확인한 뒤에 손 인식 라이브러리와 모델을 초기화한다.
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            parser.error(
                f"MediaPipe import 실패: {error}\n"
                f"실행 중인 Python: {sys.executable}\n"
                "수업용 Python 환경을 활성화한 후 다시 실행하세요. "
                "Section-Project/README.md의 설치 안내도 확인하세요."
            )

        log(f"MediaPipe import 완료; 모델 로드 시작: {args.model}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.model)),
            num_hands=1,
        )
        with vision.HandLandmarker.create_from_options(options) as detector:
            log("모델 로드 완료; 필터 반복 처리 시작")
            first_frame = True
            active_count = 0
            previous_count = None
            consecutive_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        "카메라는 열렸지만 영상 프레임을 받지 못했습니다. "
                        "카메라의 YUYV 640x480 30fps 지원 여부와 "
                        "GStreamer 경고 메시지를 확인하세요."
                    )

                # 거울처럼 좌우 반전. 필터 적용 전 원본에서 손을 인식한다.
                if first_frame:
                    log(f"모델 로드 후 프레임 수신 완료: {frame.shape}; 손 인식 시작")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if first_frame:
                    log("첫 손 인식 완료; 필터 처리 시작")
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

                # 현재 프레임의 손 크기만 사용한다. 객체 추적이나 거리(cm) 측정은 하지 않는다.
                strength = 0.0
                if hand is not None and active_count != 0:
                    strength = calculate_strength(
                        hand, frame.shape, args.far_size, args.near_size,
                    )
                output = apply_filter(frame, active_count, strength)
                # 필터 적용 후 글씨를 그려서 어떤 필터에서도 읽을 수 있게 한다.
                detected_text = str(detected_count) if hand is not None else "NO HAND"
                lines = (
                    f"Fingers: {detected_text}",
                    f"Filter: {active_count} - {FILTER_NAMES[active_count]}",
                    f"Strength: {strength:.0%}  (Near: strong / Far: weak)",
                    "1:Gray  2:Blur  3:Edge  4:Sharp  5:Hue",
                    "0 / No hand: Original    Q / ESC: Quit",
                )
                for i, text in enumerate(lines):
                    position = (10, 25 + i * 25)
                    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(output, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # 막대 전체 길이가 100%, 채워진 부분이 현재 강도.
                cv2.rectangle(output, (10, 140), (210, 154), (0, 0, 0), -1)
                if strength > 0:
                    cv2.rectangle(output, (10, 140), (10 + round(200 * strength), 154),
                                  (0, 255, 255), -1)
                cv2.rectangle(output, (10, 140), (210, 154), (255, 255, 255), 1)

                if first_frame:
                    log("첫 필터 처리 완료; 영상 창 표시 시작")
                cv2.imshow("Finger Camera Filters", output)
                key = cv2.waitKey(1) & 0xFF
                if first_frame:
                    log("영상 창 표시 완료; 반복 처리 중 (종료: q 또는 ESC)")
                    first_frame = False
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
