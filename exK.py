import cv2
import numpy as np


# USB 카메라의 YUYV 포맷은 GStreamer에서 YUY2로 지정합니다.
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def main():
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 상태: [x, y, vx, vy], 측정: [x, y]
    # 속도 단위는 픽셀/프레임입니다.
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    kalman.processNoiseCov = np.diag(
        [1e-2, 1e-2, 5e-2, 5e-2]
    ).astype(np.float32)
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    found = False
    last_radius = 0
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "카메라를 열 수 없습니다. /dev/video0의 YUYV "
                "640x480 30fps 지원 여부와 OpenCV GStreamer 지원을 확인하세요."
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # K: 전처리 및 초록색 영역 추출
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            mask = cv2.inRange(lab, green_lower, green_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            center = None

            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 500:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    moments = cv2.moments(contour)
                    if moments["m00"] > 0:
                        center = (
                            int(moments["m10"] / moments["m00"]),
                            int(moments["m01"] / moments["m00"]),
                        )
                        last_radius = int(np.ceil(radius))
                        # 실제 검출: 파란 외접원과 무게중심
                        cv2.circle(
                            frame, (round(x), round(y)),
                            last_radius, (255, 0, 0), 2,
                        )
                        cv2.circle(frame, center, 5, (255, 0, 0), -1)

            # L: 최초 검출 위치로 필터 초기화
            if center is not None and not found:
                initial_state = np.array([
                    [center[0]], [center[1]], [0], [0],
                ], dtype=np.float32)
                kalman.statePre = initial_state.copy()
                kalman.statePost = initial_state.copy()
                found = True

            if found:
                predicted_state = kalman.predict()
                if center is not None:
                    measurement = np.array([
                        [center[0]], [center[1]],
                    ], dtype=np.float32)
                    estimated_state = kalman.correct(measurement)
                    status = "Tracking"
                else:
                    # 객체가 가려지면 측정값 없이 예측만 수행
                    estimated_state = predicted_state
                    status = "Predicting"

                estimated_center = (
                    int(round(float(estimated_state[0, 0]))),
                    int(round(float(estimated_state[1, 0]))),
                )
                # 필터 추정: 노란 원, 반지름은 마지막 검출값 유지
                cv2.circle(
                    frame, estimated_center, last_radius, (0, 255, 255), 2,
                )
                cv2.circle(frame, estimated_center, 3, (0, 255, 255), -1)
                label = f"{status}: {estimated_center}"
            else:
                label = "Waiting for green object"

            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2,
            )
            cv2.imshow("Object Detection and Tracking", frame)
            cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
