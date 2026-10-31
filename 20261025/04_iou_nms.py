#!/usr/bin/env python3
"""04_DL-Object-Detection.ipynb: IoU와 confidence filtering, NMS 구현.

실행: python3 20261025/04_iou_nms.py
좌표는 (x1, y1, x2, y2), 면적은 (x2-x1)*(y2-y1) 기준입니다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def convert_bbox_coord(bbox):
    """OpenCV의 (x, y, w, h)를 (x1, y1, x2, y2)로 변환."""
    x, y, width, height = bbox
    if width < 0 or height < 0:
        raise ValueError("너비와 높이는 음수일 수 없습니다.")
    return (x, y, x + width, y + height)


def calculate_iou(bbox_a, bbox_b):
    """두 박스의 교집합/합집합. 면적이 0인 박스도 처리합니다."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    intersection_area = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def validate_inputs(boxes, scores, threshold):
    if len(boxes) != len(scores):
        raise ValueError("boxes와 scores의 길이가 같아야 합니다.")
    if not 0 <= threshold <= 1:
        raise ValueError("임계값은 0~1이어야 합니다.")
    for box, score in zip(boxes, scores):
        if len(box) != 4 or not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError("박스 좌표와 점수는 유한한 숫자여야 합니다.")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("박스의 끝 좌표는 시작 좌표 이상이어야 합니다.")


def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """최소 confidence 이상인 박스와 점수만 유지."""
    validate_inputs(boxes, scores, conf_threshold)
    indices = [i for i, score in enumerate(scores) if score >= conf_threshold]
    return [boxes[i] for i in indices], [scores[i] for i in indices]


def apply_nms(boxes, scores, iou_threshold=0.5):
    """점수가 높은 박스를 선택하고 IoU가 임계값보다 큰 나머지 박스 제거.

    이 함수는 단일 클래스 기준입니다. 여러 클래스는 클래스별로 호출합니다.
    confidence filtering은 prune_low_conf()로 먼저 적용합니다.
    """
    validate_inputs(boxes, scores, iou_threshold)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable").tolist()
    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order if calculate_iou(boxes[best], boxes[i]) <= iou_threshold]
    return [boxes[i] for i in kept], [scores[i] for i in kept]


def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """OpenCV 구현과 비교. OpenCV는 confidence가 임계값과 같아도 제거합니다."""
    validate_inputs(boxes, scores, conf_threshold)
    validate_inputs(boxes, scores, iou_threshold)
    if len(boxes) == 0:
        return [], []
    xywh = [[float(x1), float(y1), float(x2-x1), float(y2-y1)]
            for x1, y1, x2, y2 in boxes]
    kept = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], conf_threshold, iou_threshold)
    kept = np.asarray(kept).reshape(-1).tolist()
    return [boxes[i] for i in kept], [scores[i] for i in kept]


def draw_boxes(image, boxes, scores):
    result = image.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"{score:.2f}", (x1, max(18, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return result


def get_contour(img_bgr):
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([25, 10, 30])
    green_upper = np.array([100, 255, 255])

    img_blr = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    blr_hsv = cv2.cvtColor(img_blr, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(blr_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(blr_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red, cntr_grn = None, None

    if len(contours_red) > 0:
        cntr_red = max(contours_red, key=cv2.contourArea)
    if len(contours_grn) > 0:
        cntr_grn = max(contours_grn, key=cv2.contourArea)

    return cntr_red, cntr_grn

def main():
    parser = argparse.ArgumentParser(description="IoU, NMS 직접 구현 및 OpenCV 비교")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/iou_nms")
    args = parser.parse_args()
    if not (0 <= args.conf_threshold <= 1 and 0 <= args.iou_threshold <= 1):
        parser.error("임계값은 0~1이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        path = ROOT / f"src/images/two_apples{i}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        red, green = get_contour(image)
        if red is None or green is None:
            raise RuntimeError(f"빨강/초록 윤곽선 검출 실패: {path}")
        boxes = [convert_bbox_coord(cv2.boundingRect(contour)) for contour in (red, green)]
        iou = calculate_iou(*boxes)
        print(f"{path.name}: IoU = {iou:.4f}")
        result = draw_boxes(image, boxes, [1.0, 1.0])
        cv2.putText(result, f"IoU: {iou:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if not cv2.imwrite(str(args.output_dir / path.name), result):
            raise OSError("IoU 이미지 저장 실패")

    # 실행마다 같은 박스로 NMS 단계별 결과를 재현합니다.
    boxes = [(50, 50, 250, 250), (60, 60, 255, 255), (300, 80, 500, 280),
             (310, 90, 510, 290), (100, 300, 220, 450)]
    scores = [0.95, 0.85, 0.90, 0.70, 0.30]
    filtered_boxes, filtered_scores = prune_low_conf(boxes, scores, args.conf_threshold)
    kept_boxes, kept_scores = apply_nms(filtered_boxes, filtered_scores, args.iou_threshold)
    cv_boxes, cv_scores = apply_nms_cv2(boxes, scores, args.conf_threshold, args.iou_threshold)
    print(f"NMS: {len(boxes)} → confidence {len(filtered_boxes)} → NMS {len(kept_boxes)}")
    print(f"Custom: {kept_boxes}, {kept_scores}")
    print(f"OpenCV: {cv_boxes}, {cv_scores}")
    canvas = np.zeros((480, 560, 3), dtype=np.uint8)
    comparison = np.hstack([draw_boxes(canvas, boxes, scores),
                            draw_boxes(canvas, kept_boxes, kept_scores),
                            draw_boxes(canvas, cv_boxes, cv_scores)])
    if not cv2.imwrite(str(args.output_dir / "nms_before_custom_opencv.png"), comparison):
        raise OSError("NMS 이미지 저장 실패")
    print(f"결과 이미지: {args.output_dir}")


if __name__ == "__main__":
    main()
