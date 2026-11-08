"""모델 없이 설정된 연결 방식으로 카메라만 확인한다. 종료는 q."""

import time

from main import load_config, memory_summary, open_camera


def main():
    import cv2

    cap = open_camera(cv2, load_config())
    print(f"카메라만 확인합니다. {memory_summary()}", flush=True)
    last_report = time.perf_counter()
    frames = 0
    try:
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. 장치와 camera_backend를 확인하세요.")
        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            elapsed = time.perf_counter() - started
            if elapsed >= 1.0:
                print(f"카메라 수신 지연: {elapsed:.1f}초 / {memory_summary()}", flush=True)
            if not ok or frame is None or frame.size == 0:
                print("프레임 수신 실패", flush=True)
                break
            frames += 1
            if frame.shape[:2] != (240, 320):
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.imshow("Camera check - q: quit", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            now = time.perf_counter()
            if now - last_report >= 5:
                print(f"최근 수신: {frames / (now - last_report):.1f} FPS / {memory_summary()}", flush=True)
                last_report = now
                frames = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("카메라 진단을 종료합니다.")
