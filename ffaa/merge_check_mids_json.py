import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from get_label import get_label_all


LABEL_MAP = {
    0: "real",
    1: "pad",
    2: "deepfake",
    3: "makeup",
}
VALID_ANSWER_LABELS = {0, 1, 2, 3}
QUALITY_PATTERN = re.compile(r"\bquality\s*-\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


class ReservoirSampler:
    def __init__(self, max_size: int, rng: random.Random) -> None:
        self.max_size = max_size
        self.rng = rng
        self.items: List[Dict[str, Any]] = []
        self.seen = 0

    def add(self, sample: Dict[str, Any]) -> None:
        self.seen += 1
        if len(self.items) < self.max_size:
            self.items.append(sample)
            return

        idx = self.rng.randint(0, self.seen - 1)
        if idx < self.max_size:
            self.items[idx] = sample


def iter_json_items(json_path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()

    with json_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        while first and (first.isspace() or first == "\ufeff"):
            first = f.read(1)

        if not first:
            return

        if first != "[":
            payload = first + f.read()
            obj = json.loads(payload)
            if isinstance(obj, list):
                for item in obj:
                    yield item
            else:
                yield obj
            return

        buffer = ""
        eof = False
        while True:
            if not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            pos = 0
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1

                if pos < len(buffer) and buffer[pos] == "]":
                    return

                if pos >= len(buffer):
                    break

                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break

                yield item
                pos = end

            if pos > 0:
                buffer = buffer[pos:]

            if eof:
                tail = buffer.strip()
                if tail == "" or tail == "]":
                    return
                raise ValueError(f"Malformed JSON in {json_path}: trailing data starts with {tail[:80]!r}")


def get_difficulty(sample: Dict[str, Any]) -> Optional[str]:
    answers = sample.get("answers")
    if not isinstance(answers, list):
        return None

    results = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        result = str(ans.get("result", "")).strip().lower()
        if "real" in result:
            results.append("real")
        elif "fake" in result:
            results.append("fake")
        elif "makeup" in result:
            results.append("fake")

    if len(results) < 2:
        return None
    if len(set(results)) == 1:
        return "easy"
    return "hard"


def count_answer_labels(
    sample: Dict[str, Any],
    label_counter: Counter,
    stats_counter: Optional[Counter] = None,
    stats_prefix: str = "answer",
) -> None:
    answers = sample.get("answers")
    if not isinstance(answers, list):
        if stats_counter is not None:
            stats_counter[f"{stats_prefix}_not_list"] += 1
        return

    for ans in answers:
        if not isinstance(ans, dict):
            if stats_counter is not None:
                stats_counter[f"{stats_prefix}_not_dict"] += 1
            continue

        if "label" not in ans:
            if stats_counter is not None:
                stats_counter[f"{stats_prefix}_label_missing"] += 1
            continue

        raw_label = ans.get("label")
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            if stats_counter is not None:
                stats_counter[f"{stats_prefix}_label_invalid"] += 1
            continue

        if label in VALID_ANSWER_LABELS:
            label_counter[label] += 1
        else:
            if stats_counter is not None:
                stats_counter[f"{stats_prefix}_label_out_of_range"] += 1


def parse_cls_label(sample: Dict[str, Any]) -> Optional[int]:
    raw = sample.get("cls_label")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def extract_quality_from_content(content: str) -> Optional[float]:
    if not isinstance(content, str):
        return None
    match = QUALITY_PATTERN.search(content)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_sample_quality(sample: Dict[str, Any]) -> Optional[float]:
    answers = sample.get("answers")
    if not isinstance(answers, list):
        return None

    qualities: List[float] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        quality = extract_quality_from_content(ans.get("content", ""))
        if quality is not None:
            qualities.append(quality)

    if not qualities:
        return None
    return min(qualities)


def should_filter_low_quality_real(
    sample: Dict[str, Any],
    quality_threshold: float,
) -> bool:
    cls_label = parse_cls_label(sample)
    if cls_label != 0:
        return False

    quality = extract_sample_quality(sample)
    if quality is None:
        return True
    return quality < quality_threshold


def build_selected_by_class(
    forced_by_class: Dict[str, List[Dict[str, Any]]],
    samplers: Dict[str, ReservoirSampler],
    sample_per_class: int,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    selected_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for cls_name in ("real", "pad", "deepfake", "makeup"):
        forced_items = forced_by_class[cls_name]
        sampled_items = list(samplers[cls_name].items)
        rng.shuffle(sampled_items)

        remaining_slots = sample_per_class - len(forced_items)
        if remaining_slots <= 0:
            selected = forced_items
        else:
            selected = forced_items + sampled_items[:remaining_slots]
        selected_by_class[cls_name] = selected
    return selected_by_class


def build_train_eval_split(
    selected_by_class: Dict[str, List[Dict[str, Any]]],
    train_ratio: float,
    rng: random.Random,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    balanced_selected: List[Dict[str, Any]] = []
    for cls_name in ("real", "pad", "deepfake", "makeup"):
        balanced_selected.extend(selected_by_class[cls_name])

    rng.shuffle(balanced_selected)
    split_index = math.floor(len(balanced_selected) * train_ratio)
    train_data = balanced_selected[:]
    eval_data = balanced_selected[split_index:]
    return balanced_selected, train_data, eval_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse merged MIDS JSON files, split easy/hard by answers, sample hard data "
            "per class, and write train/eval json files."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/merged",
        help="Directory containing merged json files.",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json",
        help="Path to train json output.",
    )
    parser.add_argument(
        "--output_eval",
        type=str,
        default="/datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json",
        help="Path to eval json output.",
    )
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=800000,
        # default=12900,
        help="How many hard samples to select for each class.",
    )
    parser.add_argument(
        "--always_include_files",
        nargs="*",
        default=["mids_dir_err2.json,mids_dir_err3.json"],
        # default=[""],
        help=(
            "JSON filenames to include fully (hard samples) without random selection. "
            "You can pass multiple names: --always_include_files a.json b.json"
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Train split ratio for sampled data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="For debugging: stop after processing this many items (0 means all).",
    )
    parser.add_argument(
        "--real_quality_threshold",
        type=float,
        default=0.4,
        help=(
            "If cls_label==0 (real) and extracted quality is below this threshold, "
            "remove the sample. Missing quality is also removed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_train = Path(args.output_train)
    output_eval = Path(args.output_eval)
    rng = random.Random(args.seed)
    always_include_set = set()
    for token in args.always_include_files:
        for part in token.split(","):
            item = part.strip()
            if item:
                always_include_set.add(item)

    if args.sample_per_class <= 0:
        raise ValueError("--sample_per_class must be > 0")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be between 0 and 1")
    if args.real_quality_threshold < 0.0:
        raise ValueError("--real_quality_threshold must be >= 0")

    json_files = sorted(
        p for p in input_dir.glob("*.json") if p.is_file()
    )
    if not json_files:
        print(f"No json files found in: {input_dir}")
        return

    samplers = {
        cls_name: ReservoirSampler(args.sample_per_class, rng)
        for cls_name in LABEL_MAP.values()
    }
    forced_by_class = {cls_name: [] for cls_name in LABEL_MAP.values()}
    total_stats = Counter()
    hard_count_by_class = Counter()
    forced_count_by_class = Counter()
    forced_files_found = set()
    all_answer_label_counts = Counter()
    hard_answer_label_counts = Counter()

    for json_file in json_files:
        file_stats = Counter()
        is_forced_file = (
            json_file.name in always_include_set or str(json_file) in always_include_set
        )
        if is_forced_file:
            forced_files_found.add(json_file.name)
        try:
            for sample in iter_json_items(json_file):
                if args.max_items > 0 and total_stats["processed"] >= args.max_items:
                    break

                total_stats["processed"] += 1
                file_stats["processed"] += 1

                if not isinstance(sample, dict):
                    total_stats["invalid_sample"] += 1
                    file_stats["invalid_sample"] += 1
                    continue

                # if should_filter_low_quality_real(
                #     sample=sample,
                #     quality_threshold=args.real_quality_threshold,
                # ):
                #     total_stats["real_low_quality_removed"] += 1
                #     file_stats["real_low_quality_removed"] += 1
                #     continue

                count_answer_labels(
                    sample,
                    all_answer_label_counts,
                    stats_counter=total_stats,
                    stats_prefix="all_answers",
                )

                difficulty = get_difficulty(sample)
                if difficulty is None:
                    total_stats["invalid_answers"] += 1
                    file_stats["invalid_answers"] += 1
                    continue

                total_stats[difficulty] += 1
                file_stats[difficulty] += 1

                if difficulty != "hard":
                    continue

                count_answer_labels(
                    sample,
                    hard_answer_label_counts,
                    stats_counter=total_stats,
                    stats_prefix="hard_answers",
                )

                image_path = sample.get("image")
                if not isinstance(image_path, str) or image_path == "":
                    total_stats["missing_image"] += 1
                    file_stats["missing_image"] += 1
                    continue

                label_id = get_label_all(image_path)
                cls_name = LABEL_MAP.get(label_id)
                if cls_name is None:
                    total_stats["unknown_label"] += 1
                    file_stats["unknown_label"] += 1
                    continue

                hard_count_by_class[cls_name] += 1
                total_stats["hard_valid"] += 1
                file_stats["hard_valid"] += 1
                if is_forced_file:
                    forced_by_class[cls_name].append(sample)
                    forced_count_by_class[cls_name] += 1
                    total_stats["forced_kept"] += 1
                    file_stats["forced_kept"] += 1
                else:
                    samplers[cls_name].add(sample)

        except Exception as e:
            print(f"Error while processing {json_file}: {e}")
            total_stats["file_error"] += 1
            file_stats["file_error"] += 1

        print(
            f"[{json_file.name}] "
            f"processed={file_stats['processed']} "
            f"easy={file_stats['easy']} "
            f"hard={file_stats['hard']} "
            f"hard_valid={file_stats['hard_valid']} "
            f"real_low_quality_removed={file_stats['real_low_quality_removed']} "
            f"forced_kept={file_stats['forced_kept']} "
            f"invalid_answers={file_stats['invalid_answers']} "
            f"unknown_label={file_stats['unknown_label']}"
        )

        if args.max_items > 0 and total_stats["processed"] >= args.max_items:
            print(f"Reached max_items={args.max_items}. Stopping early.")
            break

    selected_by_class = build_selected_by_class(
        forced_by_class=forced_by_class,
        samplers=samplers,
        sample_per_class=args.sample_per_class,
        rng=rng,
    )

    balanced_selected, train_data, eval_data = build_train_eval_split(
        selected_by_class=selected_by_class,
        train_ratio=args.train_ratio,
        rng=rng,
    )

    selected_answer_label_counts = Counter()
    for sample in balanced_selected:
        if isinstance(sample, dict):
            count_answer_labels(sample, selected_answer_label_counts)

    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    with output_train.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with output_eval.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"input_dir: {input_dir}")
    print(f"processed: {total_stats['processed']}")
    print(f"easy: {total_stats['easy']}")
    print(f"hard: {total_stats['hard']}")
    print(f"hard_valid: {total_stats['hard_valid']}")
    print(f"real_low_quality_removed: {total_stats['real_low_quality_removed']}")
    print(f"forced_kept: {total_stats['forced_kept']}")
    print(f"invalid_sample: {total_stats['invalid_sample']}")
    print(f"invalid_answers: {total_stats['invalid_answers']}")
    print(f"missing_image: {total_stats['missing_image']}")
    print(f"unknown_label: {total_stats['unknown_label']}")
    print(f"file_error: {total_stats['file_error']}")
    print(
        f"all_answer_label_count: "
        f"0={all_answer_label_counts[0]}, 1={all_answer_label_counts[1]}, "
        f"2={all_answer_label_counts[2]}, 3={all_answer_label_counts[3]}"
    )
    print(
        f"hard_answer_label_count: "
        f"0={hard_answer_label_counts[0]}, 1={hard_answer_label_counts[1]}, "
        f"2={hard_answer_label_counts[2]}, 3={hard_answer_label_counts[3]}"
    )
    print(
        f"selected_answer_label_count: "
        f"0={selected_answer_label_counts[0]}, 1={selected_answer_label_counts[1]}, "
        f"2={selected_answer_label_counts[2]}, 3={selected_answer_label_counts[3]}"
    )
    print(
        f"all_answer_label_issues: "
        f"not_list={total_stats['all_answers_not_list']}, "
        f"not_dict={total_stats['all_answers_not_dict']}, "
        f"missing={total_stats['all_answers_label_missing']}, "
        f"invalid={total_stats['all_answers_label_invalid']}, "
        f"out_of_range={total_stats['all_answers_label_out_of_range']}"
    )
    print(
        f"hard_answer_label_issues: "
        f"not_list={total_stats['hard_answers_not_list']}, "
        f"not_dict={total_stats['hard_answers_not_dict']}, "
        f"missing={total_stats['hard_answers_label_missing']}, "
        f"invalid={total_stats['hard_answers_label_invalid']}, "
        f"out_of_range={total_stats['hard_answers_label_out_of_range']}"
    )
    print()

    for cls_name in ("real", "pad", "deepfake", "makeup"):
        print(
            f"{cls_name}: hard={hard_count_by_class[cls_name]}, "
            f"forced={forced_count_by_class[cls_name]}, "
            f"selected={len(selected_by_class[cls_name])}"
        )

    missing_forced = sorted(always_include_set - forced_files_found)
    print()
    print(f"always_include_files: {sorted(always_include_set)}")
    print(f"always_include_files_found: {sorted(forced_files_found)}")
    if missing_forced:
        print(f"always_include_files_missing: {missing_forced}")

    for cls_name in ("real", "pad", "deepfake", "makeup"):
        overflow = len(forced_by_class[cls_name]) - args.sample_per_class
        if overflow > 0:
            print(
                f"warning: {cls_name} forced samples exceed target by {overflow}; "
                f"selected count kept above target to preserve forced inclusion."
            )

    print()
    print(f"real_quality_threshold: {args.real_quality_threshold}")
    print("real quality rule: remove cls_label==0 if quality<threshold or quality missing")
    print()
    print(f"sample_per_class target: {args.sample_per_class}")
    print(f"total selected: {len(balanced_selected)}")
    print(f"train: {len(train_data)} -> {output_train}")
    print(f"eval: {len(eval_data)} -> {output_eval}")


if __name__ == "__main__":
    main()
