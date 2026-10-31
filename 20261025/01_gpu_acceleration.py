#!/usr/bin/env python3
"""03_DL-and-GPU.ipynb의 2026-10-25 실습을 독립 실행 파일로 정리."""
import argparse
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

import cv2

# CPU 실행에서는 CuPy 설치가 필요하지 않습니다.
cp = None

def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output

def motion_trail_np(image, weights):
    """NumPy를 이용해 출력 픽셀을 하나씩 계산하여 오른쪽 잔상을 생성합니다."""
    image_height, image_width = image.shape
    output = np.zeros_like(image, dtype=np.float32)

    for output_y in range(image_height):
        for output_x in range(image_width):
            # 이미지 왼쪽 경계를 벗어나는 픽셀은 누적하지 않습니다.
            for shift in range(min(len(weights), output_x + 1)):
                input_x = output_x - shift
                output[output_y, output_x] += image[output_y, input_x] * weights[shift]

    return output

def motion_trail_cp(image, weights):
    """이동 거리는 순차 반복하고, 각 단계의 전체 픽셀은 GPU에서 병렬 처리합니다."""
    image_height, image_width = image.shape
    output = cp.zeros((image_height, image_width), dtype=cp.float32)

    # 워밍업처럼 짧은 가중치 배열을 전달해도 같은 함수를 사용할 수 있습니다.
    for shift in range(min(len(weights), image_width)):
        input_region = image[:, :image_width - shift]
        output_region = output[:, shift:]
        output_region += input_region * weights[shift]

    return output

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """
    kernel_height, kernel_width = kernel.shape
    padded_image = cp.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 각 출력 위치에 대응하는 입력 픽셀을 stride 간격으로 선택
            input_region = padded_image[
                kernel_y:kernel_y + output_height * stride:stride,
                kernel_x:kernel_x + output_width * stride:stride,
            ]
            output += input_region * kernel[kernel_y, kernel_x]

    output += bias
    return output

# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)


def main():
    global cp
    parser = argparse.ArgumentParser(description="NumPy/CuPy 잔상 및 Convolution2D 비교")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image", type=Path, default=ROOT / "src/images/seagull.jpg")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trail-length", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/gpu_acceleration")
    args = parser.parse_args()
    if min(args.width, args.height, args.trail_length) < 1:
        parser.error("이미지 크기와 잔상 길이는 양수여야 합니다.")
    if args.device == "cuda":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("CUDA GPU가 없습니다.")
        except ImportError as exc:
            raise SystemExit("CUDA에 맞는 CuPy를 설치하거나 --device cpu를 사용하세요.") from exc
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(args.image)
    image = cv2.resize(image, (args.width, args.height)).astype(np.float32) / 255.0
    weights = np.linspace(1.0, 0.05, args.trail_length, dtype=np.float32)
    weights /= weights.sum()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(name, array):
        path = args.output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), np.clip(array * 255, 0, 255).astype(np.uint8)):
            raise OSError(f"이미지 저장 실패: {path}")

    save("original", image)
    jobs = [
        ("motion_trail", lambda x, w: motion_trail_np(x, w),
         lambda x, w: motion_trail_cp(x, w), weights),
        ("convolution_dx", lambda x, k: np.maximum(0, convolution2d(x, k, -0.2, padding=1)),
         lambda x, k: cp.maximum(0, convolution2d_cp(x, k, -0.2, padding=1)), dx_edge_filter),
        ("convolution_dy", lambda x, k: np.maximum(0, convolution2d(x, k, -0.2, padding=1)),
         lambda x, k: cp.maximum(0, convolution2d_cp(x, k, -0.2, padding=1)), dy_edge_filter),
    ]
    for name, cpu_func, gpu_func, parameter in jobs:
        start = time.perf_counter()
        cpu_result = cpu_func(image, parameter)
        cpu_time = time.perf_counter() - start
        save(name + "_cpu", cpu_result)
        print(f"{name}: CPU {cpu_time:.6f}s")
        if args.device == "cpu":
            continue
        # CUDA 초기화/JIT 컴파일은 측정에서 제외합니다.
        gpu_func(cp.zeros((64, 64), dtype=cp.float32), cp.asarray(parameter))
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        image_cp, parameter_cp = cp.asarray(image), cp.asarray(parameter)
        cp.cuda.Stream.null.synchronize()
        upload_time = time.perf_counter() - start
        start = time.perf_counter()
        result_cp = gpu_func(image_cp, parameter_cp)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - start
        start = time.perf_counter()
        result = cp.asnumpy(result_cp)
        download_time = time.perf_counter() - start
        np.testing.assert_allclose(result, cpu_result, rtol=1e-4, atol=1e-5)
        save(name + "_gpu", result)
        total = upload_time + gpu_time + download_time
        print(f"  Upload {upload_time:.6f}s | GPU {gpu_time:.6f}s | Download {download_time:.6f}s")
        print(f"  Compute speedup {cpu_time / gpu_time:.2f}x | Including transfers {cpu_time / total:.2f}x")
    print(f"결과 이미지: {args.output_dir}")


if __name__ == "__main__":
    main()
