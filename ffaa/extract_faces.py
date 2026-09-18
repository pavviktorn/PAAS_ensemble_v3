#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_SRC_ROOT = Path("/datasets/datasets/PAD/dataset/CelebA-Spoof/Data")
DEFAULT_OUTPUT_DIR = Path("/datasets/work/vLLM/data/CelebA/real")
DEFAULT_SCRFD_MODEL = Path("/home/ubuntu/.insightface/models/buffalo_l/det_10g.onnx")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NUMBERED_NAME = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
LIVE_LABEL = "live"


@dataclass
class Stats:
    scanned_live_images: int = 0
    read_errors: int = 0
    detect_errors: int = 0
    no_face: int = 0
    too_small: int = 0
    save_errors: int = 0
    saved: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export images from CelebA-Spoof live subfolders whose detected SCRFD face "
            "size meets a minimum threshold, and save them as sequential JPEG files."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help=f"Dataset root to scan. Default: {DEFAULT_SRC_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for extracted JPEGs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--scrfd-model",
        type=Path,
        default=DEFAULT_SCRFD_MODEL,
        help=f"SCRFD ONNX model path. Default: {DEFAULT_SCRFD_MODEL}",
    )
    parser.add_argument(
        "--det-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 640),
        help="SCRFD detector input size. Default: 640 640",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.5,
        help="SCRFD confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.4,
        help="SCRFD NMS threshold. Default: 0.4",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=None,
        help="Set to -1 for CPU only, 0 for first GPU. Default: auto",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=256.0,
        help="Minimum detected face width in pixels. Default: 256",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=256.0,
        help="Minimum detected face height in pixels. Default: 256",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=2.0,
        help="Crop size relative to the detected face size. Default: 2.0",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Zero-padding width for output filenames. Default: 6",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "Override the starting output index. By default, the script resumes after "
            "the highest numbered JPG already in the output directory."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after saving this many images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report matches without writing any files.",
    )
    return parser.parse_args()


def load_scrfd_class():
    try:
        from insightface.model_zoo.scrfd import SCRFD

        return SCRFD
    except Exception:
        spec = importlib.util.find_spec("insightface")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("insightface is not installed")

        module_path = Path(spec.submodule_search_locations[0]) / "model_zoo" / "scrfd.py"
        if not module_path.exists():
            raise RuntimeError(f"Could not locate SCRFD module: {module_path}")

        scrfd_spec = importlib.util.spec_from_file_location("_insightface_scrfd", module_path)
        if scrfd_spec is None or scrfd_spec.loader is None:
            raise RuntimeError(f"Could not load SCRFD module: {module_path}")

        module = importlib.util.module_from_spec(scrfd_spec)
        scrfd_spec.loader.exec_module(module)
        return module.SCRFD


def choose_providers(ctx_id: int | None) -> list[str]:
    available = ort.get_available_providers()
    if ctx_id is not None and ctx_id < 0:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is not available in onnxruntime")
        return ["CPUExecutionProvider"]

    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if providers:
        return providers
    if not available:
        raise RuntimeError("No onnxruntime execution providers are available")
    return available


def build_detector(
    model_path: Path,
    det_size: tuple[int, int],
    det_thresh: float,
    nms_thresh: float,
    ctx_id: int | None,
):
    SCRFD = load_scrfd_class()
    providers = choose_providers(ctx_id)
    session = ort.InferenceSession(str(model_path), providers=providers)
    detector = SCRFD(model_file=str(model_path), session=session)
    effective_ctx_id = ctx_id
    if effective_ctx_id is None:
        effective_ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    detector.prepare(
        ctx_id=effective_ctx_id,
        input_size=det_size,
        det_thresh=det_thresh,
        nms_thresh=nms_thresh,
    )
    return detector, providers, effective_ctx_id


def iter_live_dirs(src_root: Path):
    if src_root.is_dir() and src_root.name == LIVE_LABEL:
        yield src_root
        return

    for path in sorted(src_root.rglob(LIVE_LABEL)):
        if path.is_dir():
            yield path


def iter_images(src_root: Path):
    for live_dir in iter_live_dirs(src_root):
        for path in sorted(live_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def next_index(output_dir: Path) -> int:
    max_index = 0
    for existing in output_dir.glob("*.jpg"):
        match = NUMBERED_NAME.match(existing.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1 if max_index else 1


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def select_face_bbox(
    detector,
    image: np.ndarray,
    min_width: float,
    min_height: float,
) -> tuple[str, np.ndarray | None]:
    det, _ = detector.detect(image, max_num=0)
    if det is None or det.shape[0] == 0:
        return "no_face", None

    widths = det[:, 2] - det[:, 0]
    heights = det[:, 3] - det[:, 1]
    qualified = np.where((widths >= min_width) & (heights >= min_height))[0]
    if qualified.size == 0:
        return "too_small", None

    qualified_areas = widths[qualified] * heights[qualified]
    best_index = qualified[np.argmax(qualified_areas)]
    return "qualified", det[best_index, :4]


def crop_face_region(image: np.ndarray, bbox: np.ndarray, crop_scale: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width <= 0 or face_height <= 0:
        raise ValueError("Detected face bbox is invalid")

    crop_width = face_width * crop_scale
    crop_height = face_height * crop_scale
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    left = max(0, int(np.floor(center_x - crop_width / 2.0)))
    top = max(0, int(np.floor(center_y - crop_height / 2.0)))
    right = min(image.shape[1], int(np.ceil(center_x + crop_width / 2.0)))
    bottom = min(image.shape[0], int(np.ceil(center_y + crop_height / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError("Expanded crop is empty after clipping to image bounds")

    return image[top:bottom, left:right]


def save_image(image: np.ndarray, dst_path: Path) -> None:
    ok = cv2.imwrite(str(dst_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"Failed to write image: {dst_path}")


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.scrfd_model.resolve()
    det_size = tuple(args.det_size)

    if not src_root.exists():
        raise SystemExit(f"Source root does not exist: {src_root}")
    if not model_path.exists():
        raise SystemExit(f"SCRFD model does not exist: {model_path}")
    if args.digits < 1:
        raise SystemExit("--digits must be at least 1")
    if args.crop_scale < 1.0:
        raise SystemExit("--crop-scale must be at least 1.0")
    if det_size[0] <= 0 or det_size[1] <= 0:
        raise SystemExit("--det-size values must be positive")

    detector, providers, effective_ctx_id = build_detector(
        model_path=model_path,
        det_size=det_size,
        det_thresh=args.det_thresh,
        nms_thresh=args.nms_thresh,
        ctx_id=args.ctx_id,
    )
    stats = Stats()

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    current_index = args.start_index
    if current_index is None:
        current_index = next_index(output_dir) if output_dir.exists() else 1

    for image_path in iter_images(src_root):
        stats.scanned_live_images += 1

        try:
            image = load_image(image_path)
        except Exception:
            stats.read_errors += 1
            continue

        try:
            face_status, selected_bbox = select_face_bbox(
                detector=detector,
                image=image,
                min_width=args.min_width,
                min_height=args.min_height,
            )
        except Exception:
            stats.detect_errors += 1
            continue

        if face_status == "no_face":
            stats.no_face += 1
            continue
        if face_status == "too_small":
            stats.too_small += 1
            continue

        try:
            cropped_image = crop_face_region(
                image=image,
                bbox=selected_bbox,
                crop_scale=args.crop_scale,
            )
        except Exception:
            stats.save_errors += 1
            continue

        if args.dry_run:
            stats.saved += 1
        else:
            dst_path = output_dir / f"{current_index:0{args.digits}d}.jpg"
            while dst_path.exists():
                current_index += 1
                dst_path = output_dir / f"{current_index:0{args.digits}d}.jpg"
            try:
                save_image(cropped_image, dst_path)
                print(f"Saved: {dst_path} (from {image_path})")
            except Exception:
                stats.save_errors += 1
                continue
            stats.saved += 1
            current_index += 1

        if args.max_images is not None and stats.saved >= args.max_images:
            break

    print(f"source_root: {src_root}")
    print(f"output_dir: {output_dir}")
    print(f"subfolder: {LIVE_LABEL}")
    print(f"scrfd_model: {model_path}")
    print(f"providers: {providers}")
    print(f"ctx_id: {effective_ctx_id}")
    print(f"det_size: {det_size}")
    print(f"det_thresh: {args.det_thresh}")
    print(f"nms_thresh: {args.nms_thresh}")
    print(f"crop_scale: {args.crop_scale}")
    print(f"min_face: ({args.min_width}, {args.min_height})")
    print(f"scanned_live_images: {stats.scanned_live_images}")
    print(f"read_errors: {stats.read_errors}")
    print(f"detect_errors: {stats.detect_errors}")
    print(f"no_face: {stats.no_face}")
    print(f"too_small: {stats.too_small}")
    print(f"save_errors: {stats.save_errors}")
    print(f"saved: {stats.saved}")
    if args.dry_run:
        print("dry_run: true")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
