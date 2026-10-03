#!/usr/bin/env python3
"""
Move images with strong head pose from real/ into real_exclude/.

This uses OpenCV YuNet face detection. YuNet is fast on CPU and returns five
face landmarks, which are enough for a simple yaw/pitch/roll filter.

Example:
    python remove_bad_headpose.py --dry-run
    python remove_bad_headpose.py --workers 4
    python remove_bad_headpose.py --yaw-threshold 0.30 --pitch-min 0.34 --pitch-max 0.66
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/miss_axonlabs_data/real"
DEFAULT_EXCLUDE_DIR = "/datasets/work/vLLM/data/miss_axonlabs_data/real_exclude"
DEFAULT_MODEL_CANDIDATES = (
    "/datasets/work/vLLM/data_processing/down_gs/models/face_detection_yunet_2023mar.onnx",
    "models/face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2023mar.onnx",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
FRAME_STEM_RE = re.compile(r"^(.*?)(\d+)$")


_CV2 = None
_DETECTOR = None
_CFG = None


@dataclass(frozen=True)
class WorkerConfig:
    model_path: str
    detect_size: int
    score_threshold: float
    nms_threshold: float
    top_k: int
    yaw_threshold: float
    pitch_min: float
    pitch_max: float
    roll_threshold: float
    eye_y_min: float
    eye_y_max: float
    use_pnp: bool
    pnp_yaw_threshold: float
    pnp_pitch_threshold: float


@dataclass
class ImageResult:
    path: str
    status: str
    is_bad: bool = False
    reason: str = ""
    face_score: Optional[float] = None
    yaw_score: Optional[float] = None
    pitch_ratio: Optional[float] = None
    eye_y_ratio: Optional[float] = None
    mouth_y_ratio: Optional[float] = None
    roll_deg: Optional[float] = None
    pnp_yaw_deg: Optional[float] = None
    pnp_pitch_deg: Optional[float] = None
    pnp_roll_deg: Optional[float] = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move likely heavy-head-pose face images to an exclude folder."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--exclude-dir", default=DEFAULT_EXCLUDE_DIR)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Optional specific image paths to process instead of scanning input-dir.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to face_detection_yunet_2023mar.onnx. Auto-detected by default.",
    )
    parser.add_argument("--report", default="remove_bad_headpose_report.csv")
    parser.add_argument("--detect-size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=0.34,
        help="Move when nose horizontal offset score is at least this value.",
    )
    parser.add_argument(
        "--pitch-min",
        type=float,
        default=0.34,
        help="Move when nose vertical ratio is below this value.",
    )
    parser.add_argument(
        "--pitch-max",
        type=float,
        default=0.66,
        help="Move when nose vertical ratio is above this value.",
    )
    parser.add_argument(
        "--roll-threshold",
        type=float,
        default=35.0,
        help="Move when eye-line roll angle in degrees is at least this value.",
    )
    parser.add_argument(
        "--eye-y-min",
        type=float,
        default=0.34,
        help="Move as head-up when average eye y-position in face box is below this.",
    )
    parser.add_argument(
        "--eye-y-max",
        type=float,
        default=0.50,
        help="Move as head-down when average eye y-position in face box is above this.",
    )
    parser.add_argument(
        "--use-pnp",
        dest="use_pnp",
        action="store_true",
        default=True,
        help="Also estimate rough yaw/pitch angles with solvePnP.",
    )
    parser.add_argument(
        "--no-pnp",
        dest="use_pnp",
        action="store_false",
        help="Disable rough solvePnP yaw/pitch angle check.",
    )
    parser.add_argument("--pnp-yaw-threshold", type=float, default=35.0)
    parser.add_argument("--pnp-pitch-threshold", type=float, default=18.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=4,
        help="Also select same-sequence frames within N frame numbers of a bad frame.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Process only N images.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not move.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move.")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Do not preserve input subfolders in the exclude folder.",
    )
    parser.add_argument(
        "--move-no-face",
        dest="move_no_face",
        action="store_true",
        default=True,
        help="Also move images where no face is detected. Enabled by default.",
    )
    parser.add_argument(
        "--keep-no-face",
        dest="move_no_face",
        action="store_false",
        help="Do not move images where no face is detected.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N processed images.",
    )
    return parser.parse_args()


def load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for this script. Install it with: "
            "pip install opencv-python"
        ) from exc
    return cv2


def validate_cv2_detector(cv2) -> None:
    has_detector = (
        hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create")
    ) or hasattr(cv2, "FaceDetectorYN_create")
    if not has_detector:
        raise RuntimeError(
            "This OpenCV build does not include FaceDetectorYN. "
            "Install opencv-contrib-python or a newer opencv-python build."
        )


def find_model_path(model_arg: Optional[str]) -> str:
    if model_arg:
        model_path = Path(model_arg)
        if not model_path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")
        return str(model_path)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return str(path)

    joined = "\n  ".join(DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Could not find YuNet ONNX model. Pass --model explicitly.\n"
        f"Checked:\n  {joined}"
    )


def create_yunet_detector(cv2, cfg: WorkerConfig):
    input_size = (cfg.detect_size, cfg.detect_size)
    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(
            cfg.model_path,
            "",
            input_size,
            cfg.score_threshold,
            cfg.nms_threshold,
            cfg.top_k,
        )
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(
            cfg.model_path,
            "",
            input_size,
            cfg.score_threshold,
            cfg.nms_threshold,
            cfg.top_k,
        )
    validate_cv2_detector(cv2)
    raise RuntimeError("FaceDetectorYN validation passed but detector creation failed.")


def init_worker(cfg: WorkerConfig) -> None:
    global _CV2, _DETECTOR, _CFG
    _CFG = cfg
    _CV2 = load_cv2()
    if hasattr(_CV2, "setNumThreads"):
        _CV2.setNumThreads(1)
    _DETECTOR = create_yunet_detector(_CV2, cfg)


def list_images(input_dir: Path) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTS:
                paths.append(str(Path(root) / filename))
    return sorted(paths)


def resize_for_detection(img: np.ndarray, max_side: int):
    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        return img
    scale = min(1.0, float(max_side) / float(max(height, width)))
    new_w = max(32, int(round(width * scale)))
    new_h = max(32, int(round(height * scale)))
    if new_w != width or new_h != height:
        img = _CV2.resize(img, (new_w, new_h), interpolation=_CV2.INTER_AREA)

    canvas = np.full((max_side, max_side, 3), 114, dtype=img.dtype)
    top = (max_side - new_h) // 2
    left = (max_side - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = img
    return canvas


def pick_main_face(faces: np.ndarray) -> np.ndarray:
    if faces.ndim == 1:
        return faces
    areas = faces[:, 2] * faces[:, 3]
    scores = faces[:, -1]
    return faces[int(np.argmax(areas * scores))]


def norm(point: np.ndarray) -> float:
    return float(np.linalg.norm(point))


def sorted_pair_by_x(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if float(a[0]) <= float(b[0]):
        return a, b
    return b, a


def pnp_euler_degrees(
    landmarks: Sequence[np.ndarray], image_width: int, image_height: int
) -> Optional[Tuple[float, float, float]]:
    if not hasattr(_CV2, "solvePnP"):
        return None

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    image_points = np.array(
        [left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float64
    )

    model_points = np.array(
        [
            [-32.0, -28.0, -30.0],
            [32.0, -28.0, -30.0],
            [0.0, 0.0, 0.0],
            [-26.0, 32.0, -24.0],
            [26.0, 32.0, -24.0],
        ],
        dtype=np.float64,
    )

    focal = float(image_width)
    camera_matrix = np.array(
        [
            [focal, 0.0, image_width / 2.0],
            [0.0, focal, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    flag = getattr(_CV2, "SOLVEPNP_SQPNP", getattr(_CV2, "SOLVEPNP_EPNP", 1))

    try:
        ok, rvec, tvec = _CV2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=flag,
        )
    except Exception:
        return None

    if not ok:
        return None

    rot_mat, _ = _CV2.Rodrigues(rvec)
    projection = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler = _CV2.decomposeProjectionMatrix(projection)
    pitch, yaw, roll = [float(v) for v in euler.reshape(-1)[:3]]
    return yaw, pitch, roll


def score_pose(face: np.ndarray, image_width: int, image_height: int, cfg: WorkerConfig):
    if face.shape[0] < 15:
        return ImageResult(path="", status="bad_detection", error="unexpected face output")

    box = face[:4].astype(np.float32)
    pts = face[4:14].reshape(5, 2).astype(np.float32)
    face_score = float(face[-1])

    eye_a, eye_b = pts[0], pts[1]
    mouth_a, mouth_b = pts[3], pts[4]
    left_eye, right_eye = sorted_pair_by_x(eye_a, eye_b)
    left_mouth, right_mouth = sorted_pair_by_x(mouth_a, mouth_b)
    nose = pts[2]

    eye_dist = norm(right_eye - left_eye)
    if eye_dist < 8.0:
        return ImageResult(
            path="",
            status="bad_landmarks",
            face_score=face_score,
            error="eye distance too small",
        )

    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    face_axis = mouth_mid - eye_mid
    face_axis_len_sq = float(np.dot(face_axis, face_axis))
    if face_axis_len_sq < 1.0:
        return ImageResult(
            path="",
            status="bad_landmarks",
            face_score=face_score,
            error="face vertical axis too small",
        )

    x_axis = (right_eye - left_eye) / eye_dist
    centerline_anchor = eye_mid * 0.55 + mouth_mid * 0.45
    yaw_offset = abs(float(np.dot(nose - centerline_anchor, x_axis))) / eye_dist

    left_nose_dist = norm(nose - left_eye)
    right_nose_dist = norm(nose - right_eye)
    yaw_ratio = abs(left_nose_dist - right_nose_dist) / max(
        left_nose_dist + right_nose_dist, 1.0
    )
    yaw_score = max(yaw_offset, yaw_ratio)

    pitch_ratio = float(np.dot(nose - eye_mid, face_axis) / face_axis_len_sq)
    eye_y_ratio = float((eye_mid[1] - box[1]) / max(float(box[3]), 1.0))
    mouth_y_ratio = float((mouth_mid[1] - box[1]) / max(float(box[3]), 1.0))
    roll_deg = math.degrees(
        math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0]))
    )

    reasons = []
    if yaw_score >= cfg.yaw_threshold:
        reasons.append("yaw")
    if pitch_ratio <= cfg.pitch_min:
        reasons.append("pitch_up")
    if pitch_ratio >= cfg.pitch_max:
        reasons.append("pitch_down")
    if eye_y_ratio <= cfg.eye_y_min:
        reasons.append("pitch_box_up")
    if eye_y_ratio >= cfg.eye_y_max:
        reasons.append("pitch_box_down")
    if abs(roll_deg) >= cfg.roll_threshold:
        reasons.append("roll")

    pnp_yaw = None
    pnp_pitch = None
    pnp_roll = None
    if cfg.use_pnp:
        pnp = pnp_euler_degrees(
            [left_eye, right_eye, nose, left_mouth, right_mouth],
            image_width,
            image_height,
        )
        if pnp is not None:
            pnp_yaw, pnp_pitch, pnp_roll = pnp
            if abs(pnp_yaw) >= cfg.pnp_yaw_threshold:
                reasons.append("pnp_yaw")
            if abs(pnp_pitch) >= cfg.pnp_pitch_threshold:
                reasons.append("pnp_pitch")
            if abs(pnp_roll) >= cfg.roll_threshold:
                reasons.append("pnp_roll")

    return ImageResult(
        path="",
        status="ok",
        is_bad=bool(reasons),
        reason=";".join(reasons),
        face_score=face_score,
        yaw_score=yaw_score,
        pitch_ratio=pitch_ratio,
        eye_y_ratio=eye_y_ratio,
        mouth_y_ratio=mouth_y_ratio,
        roll_deg=roll_deg,
        pnp_yaw_deg=pnp_yaw,
        pnp_pitch_deg=pnp_pitch,
        pnp_roll_deg=pnp_roll,
    )


def score_image(path: str) -> ImageResult:
    cfg = _CFG
    try:
        img = _CV2.imread(path, _CV2.IMREAD_COLOR)
        if img is None:
            return ImageResult(path=path, status="read_error", error="cv2.imread failed")

        img = resize_for_detection(img, cfg.detect_size)
        height, width = img.shape[:2]
        _DETECTOR.setInputSize((width, height))
        _, faces = _DETECTOR.detect(img)
        if faces is None or len(faces) == 0:
            return ImageResult(path=path, status="no_face")

        face = pick_main_face(np.asarray(faces, dtype=np.float32))
        result = score_pose(face, width, height, cfg)
        result.path = path
        return result
    except Exception as exc:
        return ImageResult(path=path, status="error", error=repr(exc))


def iter_results(paths: Sequence[str], cfg: WorkerConfig, workers: int) -> Iterable[ImageResult]:
    if workers <= 1:
        init_worker(cfg)
        for path in paths:
            yield score_image(path)
        return

    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(cfg,)
    ) as executor:
        yield from executor.map(score_image, paths, chunksize=16)


def unique_destination(dst_path: Path) -> Path:
    if not dst_path.exists():
        return dst_path
    stem = dst_path.stem
    suffix = dst_path.suffix
    parent = dst_path.parent
    for idx in range(1, 100000):
        candidate = parent / f"{stem}_{idx:05d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique destination for {dst_path}")


def build_destination(
    src_path: str,
    input_dir: Path,
    exclude_dir: Path,
    flat: bool,
    create_dirs: bool,
) -> Path:
    src = Path(src_path)
    if flat:
        dst = exclude_dir / src.name
    else:
        try:
            rel_path = src.relative_to(input_dir)
        except ValueError:
            rel_path = Path(src.name)
        dst = exclude_dir / rel_path
    if create_dirs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    return unique_destination(dst)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def append_reason(result: ImageResult, reason: str) -> None:
    reasons = [part for part in result.reason.split(";") if part]
    if reason not in reasons:
        reasons.append(reason)
    result.reason = ";".join(reasons)


def parse_frame_key(path: str, root_dir: Path):
    image_path = Path(path).resolve()
    match = FRAME_STEM_RE.match(image_path.stem)
    if not match:
        return None

    try:
        rel_parent = image_path.parent.relative_to(root_dir)
    except ValueError:
        rel_parent = Path(image_path.parent.name)

    prefix, frame_text = match.groups()
    return str(rel_parent), prefix, int(frame_text)


def add_frame_reference(index, path: str, root_dir: Path) -> None:
    parsed = parse_frame_key(path, root_dir)
    if parsed is None:
        return
    rel_parent, prefix, frame_no = parsed
    index.setdefault((rel_parent, prefix), []).append(frame_no)


def has_near_frame(frame_numbers: Sequence[int], frame_no: int, window: int) -> bool:
    pos = bisect.bisect_left(frame_numbers, frame_no)
    if pos < len(frame_numbers) and abs(frame_numbers[pos] - frame_no) <= window:
        return True
    if pos > 0 and abs(frame_numbers[pos - 1] - frame_no) <= window:
        return True
    return False


def apply_sequence_smoothing(
    results: Sequence[ImageResult],
    input_dir: Path,
    exclude_dir: Path,
    window: int,
) -> int:
    if window <= 0:
        return 0

    bad_index = {}
    for result in results:
        if result.is_bad:
            add_frame_reference(bad_index, result.path, input_dir)

    if exclude_dir.is_dir():
        for path in list_images(exclude_dir):
            add_frame_reference(bad_index, path, exclude_dir)

    for frame_numbers in bad_index.values():
        frame_numbers.sort()

    changed = 0
    for result in results:
        if result.is_bad:
            continue
        parsed = parse_frame_key(result.path, input_dir)
        if parsed is None:
            continue
        rel_parent, prefix, frame_no = parsed
        frame_numbers = bad_index.get((rel_parent, prefix))
        if frame_numbers and has_near_frame(frame_numbers, frame_no, window):
            result.is_bad = True
            append_reason(result, "near_bad_frame")
            changed += 1

    return changed


def write_report_row(writer: csv.DictWriter, result: ImageResult, destination: str, moved: bool):
    writer.writerow(
        {
            "source": result.path,
            "destination": destination,
            "moved": int(moved),
            "status": result.status,
            "selected": int(result.is_bad),
            "reason": result.reason,
            "face_score": fmt(result.face_score),
            "yaw_score": fmt(result.yaw_score),
            "pitch_ratio": fmt(result.pitch_ratio),
            "eye_y_ratio": fmt(result.eye_y_ratio),
            "mouth_y_ratio": fmt(result.mouth_y_ratio),
            "roll_deg": fmt(result.roll_deg),
            "pnp_yaw_deg": fmt(result.pnp_yaw_deg),
            "pnp_pitch_deg": fmt(result.pnp_pitch_deg),
            "pnp_roll_deg": fmt(result.pnp_roll_deg),
            "error": result.error,
        }
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    exclude_dir = Path(args.exclude_dir).resolve()
    report_path = Path(args.report).resolve()

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.detect_size < 64:
        print("--detect-size must be at least 64", file=sys.stderr)
        return 2

    try:
        model_path = find_model_path(args.model)
        cv2 = load_cv2()
        validate_cv2_detector(cv2)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cfg = WorkerConfig(
        model_path=model_path,
        detect_size=args.detect_size,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        yaw_threshold=args.yaw_threshold,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        roll_threshold=args.roll_threshold,
        eye_y_min=args.eye_y_min,
        eye_y_max=args.eye_y_max,
        use_pnp=args.use_pnp,
        pnp_yaw_threshold=args.pnp_yaw_threshold,
        pnp_pitch_threshold=args.pnp_pitch_threshold,
    )

    if args.paths:
        paths = [str(Path(path).resolve()) for path in args.paths]
    else:
        paths = list_images(input_dir)
    if args.limit > 0:
        paths = paths[: args.limit]

    print(f"Input: {input_dir}")
    print(f"Exclude: {exclude_dir}")
    print(f"YuNet model: {model_path}")
    print(f"Images: {len(paths)}")
    print(f"Dry run: {args.dry_run}")

    if not args.dry_run:
        exclude_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source",
        "destination",
        "moved",
        "status",
        "selected",
        "reason",
        "face_score",
        "yaw_score",
        "pitch_ratio",
        "eye_y_ratio",
        "mouth_y_ratio",
        "roll_deg",
        "pnp_yaw_deg",
        "pnp_pitch_deg",
        "pnp_roll_deg",
        "error",
    ]

    start = time.time()
    results: List[ImageResult] = []
    no_face_count = 0
    error_count = 0

    for idx, result in enumerate(iter_results(paths, cfg, args.workers), start=1):
        if args.move_no_face and result.status == "no_face":
            result.is_bad = True
            result.reason = "no_face"

        if result.status == "no_face":
            no_face_count += 1
        if result.status in {"read_error", "error", "bad_detection", "bad_landmarks"}:
            error_count += 1

        results.append(result)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            rate = idx / elapsed
            selected_so_far = sum(1 for item in results if item.is_bad)
            print(
                f"Processed {idx}/{len(paths)} | selected {selected_so_far} | "
                f"{rate:.1f} images/s"
            )

    smoothed_count = apply_sequence_smoothing(
        results,
        input_dir,
        exclude_dir,
        args.smooth_window,
    )
    if smoothed_count:
        print(f"Sequence smoothing selected {smoothed_count} more images")

    counts = {
        "selected": 0,
        "moved": 0,
        "copied": 0,
        "no_face": no_face_count,
        "errors": error_count,
    }

    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            destination = ""
            moved = False
            if result.is_bad:
                counts["selected"] += 1
                dst = build_destination(
                    result.path,
                    input_dir,
                    exclude_dir,
                    args.flat,
                    create_dirs=not args.dry_run,
                )
                destination = str(dst)
                if not args.dry_run:
                    if args.copy:
                        shutil.copy2(result.path, dst)
                        counts["copied"] += 1
                    else:
                        shutil.move(result.path, dst)
                        counts["moved"] += 1
                    moved = True

            write_report_row(writer, result, destination, moved)

    elapsed = time.time() - start
    action = "copied" if args.copy else "moved"
    action_count = counts["copied"] if args.copy else counts["moved"]
    print(f"Done in {elapsed:.1f}s")
    print(f"Selected: {counts['selected']}")
    print(f"{action.capitalize()}: {action_count}")
    print(f"No face: {counts['no_face']}")
    print(f"Errors: {counts['errors']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
