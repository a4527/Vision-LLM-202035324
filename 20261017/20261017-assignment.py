"""H. 농구공 마스크의 형태학적 연산, 외곽선 검출 및 선택."""

import argparse
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def detect_ball(image):
    """LAB 마스크를 정리하고 면적이 가장 큰 외곽선을 선택한다."""
    # 노트북에서 앞서 만든 mask_enhanced를 독립적으로 재현한다.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge([clahe.apply(lightness), a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    mask = cv2.inRange(
        enhanced_lab,
        np.array([70, 130, 70], dtype=np.uint8),
        np.array([180, 230, 185], dtype=np.uint8),
    )

    # 작은 전경 노이즈 제거 → 공 내부의 작은 틈 연결.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 면적이 0인 점/선은 선택 후보에서 제외한다.
    candidates = [contour for contour in contours if cv2.contourArea(contour) > 0]
    largest = max(candidates, key=cv2.contourArea) if candidates else None

    selected_mask = np.zeros_like(closed)
    if largest is not None:
        cv2.drawContours(selected_mask, [largest], -1, 255, cv2.FILLED)

    all_contours = enhanced.copy()
    cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 2)
    result = cv2.bitwise_and(enhanced, enhanced, mask=selected_mask)
    if largest is not None:
        # OpenCV는 BGR 순서이므로 빨간색은 (0, 0, 255).
        cv2.drawContours(result, [largest], -1, (0, 0, 255), 2)

    stages = {
        "01_binary_mask": mask,
        "02_opening": opened,
        "03_closing": closed,
        "04_all_contours": all_contours,
        "05_selected_mask": selected_mask,
        "06_ball_result": result,
    }
    return stages, len(contours), largest


def make_comparison(stages):
    """처리 순서대로 2행 3열 비교 이미지를 만든다."""
    tiles = []
    for name, stage in stages.items():
        tile = cv2.cvtColor(stage, cv2.COLOR_GRAY2BGR) if stage.ndim == 2 else stage
        tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(
            tile, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=SCRIPT_DIR.parent / "src/images/basketball_crop.jpg",
        help="입력 이미지 경로 (기본값: 저장소의 basketball_crop.jpg)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "outputs",
        help="결과 PNG 저장 폴더",
    )
    parser.add_argument("--show", action="store_true", help="비교 이미지를 창으로 표시")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"입력 이미지가 없습니다: {args.image}")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"이미지를 읽을 수 없습니다: {args.image}")

    stages, count, largest = detect_ball(image)
    comparison = make_comparison(stages)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, output in {**stages, "07_comparison": comparison}.items():
        path = args.output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), output):
            raise OSError(f"이미지 저장 실패: {path}")

    print(f"검출한 외곽선: {count}개")
    if largest is None:
        print("선택 가능한 외곽선이 없습니다. 빈 마스크와 결과를 저장했습니다.")
    else:
        print(f"선택 기준: 최대 면적 / 선택 면적: {cv2.contourArea(largest):.1f} px²")
    print(f"결과 저장 위치: {args.output_dir.resolve()}")

    if args.show:
        try:
            cv2.imshow("Basketball Contour Detection", comparison)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
