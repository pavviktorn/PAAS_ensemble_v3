import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from get_label import get_label_all


LABEL_MAP = {
    0: "real",
    1: "pad",
    2: "deepfake",
    3: "makeup",
}


class JsonArrayWriter:
    def __init__(self, output_path: Path, indent: Optional[int] = None) -> None:
        self.output_path = output_path
        self.indent = indent
        self.fp = None
        self._first = True
        self.count = 0

    def __enter__(self) -> "JsonArrayWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.output_path.open("w", encoding="utf-8")
        self.fp.write("[\n")
        return self

    def write(self, item: Any) -> None:
        if self.fp is None:
            raise RuntimeError("Writer is not opened.")
        if not self._first:
            self.fp.write(",\n")
        if self.indent is None:
            json.dump(item, self.fp, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(item, self.fp, ensure_ascii=False, indent=self.indent)
        self._first = False
        self.count += 1

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.fp is not None:
            self.fp.write("\n]\n")
            self.fp.close()
            self.fp = None


class ReservoirSampler:
    def __init__(self, max_size: int, rng: random.Random) -> None:
        self.max_size = max_size
        self.rng = rng
        self.items: List[Any] = []
        self.seen = 0

    def add(self, sample: Any) -> None:
        self.seen += 1
        if len(self.items) < self.max_size:
            self.items.append(sample)
            return
        idx = self.rng.randint(0, self.seen - 1)
        if idx < self.max_size:
            self.items[idx] = sample


def iter_json_items(json_path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """
    Streaming iterator over JSON payload.
    Supports:
    - list root: yields each item
    - dict/scalar root: yields single object
    """
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
                raise ValueError(
                    f"Malformed JSON in {json_path}: trailing data starts with {tail[:80]!r}"
                )


def get_prefix(filename: str) -> str:
    """
    Extract merge prefix from filename.
    Examples:
      mids_df_easy_1.json -> mids_df_easy
      mids_df_hard_7.json -> mids_df_hard
      mids_dir_err1_0.json -> mids_dir_err1
    """
    name = filename[:-5] if filename.endswith(".json") else filename
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def parse_always_include(files: List[str]) -> set:
    out = set()
    for token in files:
        for part in token.split(","):
            item = part.strip()
            if item:
                out.add(item)
    return out


def write_json(path: Path, data: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_json_files_by_prefix(input_dir: Path, merged_dir: Path) -> Counter:
    stats = Counter()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    groups = defaultdict(list)
    for p in sorted(input_dir.glob("*.json")):
        if p.is_file():
            prefix = get_prefix(p.name)
            groups[prefix].append(p)

    merged_dir.mkdir(parents=True, exist_ok=True)
    if not groups:
        print(f"[merge] No json files found in {input_dir}")
        return stats

    print(f"[merge] grouped prefixes: {len(groups)}")
    for prefix, files in sorted(groups.items()):
        out_path = merged_dir / f"{prefix}.json"
        merged_count = 0
        file_errors = 0
        with JsonArrayWriter(out_path) as writer:
            for src in sorted(files):
                try:
                    for item in iter_json_items(src):
                        writer.write(item)
                        merged_count += 1
                except Exception as e:
                    file_errors += 1
                    print(f"[merge] Error reading {src}: {e}")

        print(
            f"[merge] {prefix}: files={len(files)} merged_items={merged_count} errors={file_errors} -> {out_path}"
        )
        stats["prefixes"] += 1
        stats["files"] += len(files)
        stats["merged_items"] += merged_count
        stats["file_errors"] += file_errors

    return stats


def select_legacy_name_mode(
    merged_dir: Path,
    output_train: Path,
    output_eval: Path,
    easy_n: int,
    hard_n: int,
    train_ratio: float,
    rng: random.Random,
) -> Counter:
    """
    Legacy behavior from merge_mids_json.py:
    - if filename contains easy -> sample EASY_N
    - if filename contains hard -> sample HARD_N
    - otherwise take all
    - if filename contains real or dir -> take all
    - shuffle all_selected
    - train file keeps all_selected (legacy behavior)
    - eval gets tail split
    """
    stats = Counter()
    all_selected: List[Any] = []

    for fpath in sorted(merged_dir.glob("*.json")):
        if not fpath.is_file():
            continue
        lower_name = fpath.name.lower()

        if "easy" in lower_name:
            target_n = easy_n
        elif "hard" in lower_name:
            target_n = hard_n
        else:
            target_n = None
        if "real" in lower_name or "dir" in lower_name:
            target_n = None

        print(f"[legacy] Processing: {fpath.name}")
        selected_for_file: List[Any]
        if target_n is None:
            selected_for_file = list(iter_json_items(fpath))
            print(f"[legacy]   take all: {len(selected_for_file)}")
        else:
            sampler = ReservoirSampler(target_n, rng)
            for item in iter_json_items(fpath):
                sampler.add(item)
            selected_for_file = sampler.items
            print(
                f"[legacy]   sampled: {len(selected_for_file)} (target={target_n}, seen={sampler.seen})"
            )

        all_selected.extend(selected_for_file)
        stats["files"] += 1

    rng.shuffle(all_selected)
    split_index = math.floor(len(all_selected) * train_ratio)
    train_data = all_selected[:]  # keep old behavior
    eval_data = all_selected[split_index:]

    write_json(output_train, train_data)
    write_json(output_eval, eval_data)

    stats["selected_total"] = len(all_selected)
    stats["train"] = len(train_data)
    stats["eval"] = len(eval_data)
    return stats


def get_difficulty(sample: Dict[str, Any]) -> Optional[str]:
    answers = sample.get("answers")
    if not isinstance(answers, list):
        return None

    results = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        result = str(ans.get("result", "")).strip().lower()
        if result in {"real", "fake"}:
            results.append(result)

    if len(results) < 2:
        return None
    return "easy" if len(set(results)) == 1 else "hard"


def select_hard_class_mode(
    merged_dir: Path,
    output_train: Path,
    output_eval: Path,
    sample_per_class: int,
    always_include_files: List[str],
    train_ratio: float,
    rng: random.Random,
    max_items: int,
) -> Counter:
    stats = Counter()
    always_include_set = parse_always_include(always_include_files)

    json_files = sorted(p for p in merged_dir.glob("*.json") if p.is_file())
    if not json_files:
        print(f"[hard_class] No json files found in: {merged_dir}")
        return stats

    samplers = {
        cls_name: ReservoirSampler(sample_per_class, rng) for cls_name in LABEL_MAP.values()
    }
    forced_by_class = {cls_name: [] for cls_name in LABEL_MAP.values()}
    hard_count_by_class = Counter()
    forced_count_by_class = Counter()
    forced_files_found = set()

    for json_file in json_files:
        file_stats = Counter()
        is_forced_file = (
            json_file.name in always_include_set or str(json_file) in always_include_set
        )
        if is_forced_file:
            forced_files_found.add(json_file.name)

        try:
            for sample in iter_json_items(json_file):
                if max_items > 0 and stats["processed"] >= max_items:
                    break

                stats["processed"] += 1
                file_stats["processed"] += 1

                if not isinstance(sample, dict):
                    stats["invalid_sample"] += 1
                    file_stats["invalid_sample"] += 1
                    continue

                difficulty = get_difficulty(sample)
                if difficulty is None:
                    stats["invalid_answers"] += 1
                    file_stats["invalid_answers"] += 1
                    continue

                stats[difficulty] += 1
                file_stats[difficulty] += 1
                if difficulty != "hard":
                    continue

                image_path = sample.get("image")
                if not isinstance(image_path, str) or image_path == "":
                    stats["missing_image"] += 1
                    file_stats["missing_image"] += 1
                    continue

                label_id = get_label_all(image_path)
                cls_name = LABEL_MAP.get(label_id)
                if cls_name is None:
                    stats["unknown_label"] += 1
                    file_stats["unknown_label"] += 1
                    continue

                hard_count_by_class[cls_name] += 1
                stats["hard_valid"] += 1
                file_stats["hard_valid"] += 1

                if is_forced_file:
                    forced_by_class[cls_name].append(sample)
                    forced_count_by_class[cls_name] += 1
                    stats["forced_kept"] += 1
                    file_stats["forced_kept"] += 1
                else:
                    samplers[cls_name].add(sample)

        except Exception as e:
            stats["file_error"] += 1
            file_stats["file_error"] += 1
            print(f"[hard_class] Error while processing {json_file}: {e}")

        print(
            f"[hard_class:{json_file.name}] "
            f"processed={file_stats['processed']} "
            f"easy={file_stats['easy']} "
            f"hard={file_stats['hard']} "
            f"hard_valid={file_stats['hard_valid']} "
            f"forced_kept={file_stats['forced_kept']} "
            f"invalid_answers={file_stats['invalid_answers']} "
            f"unknown_label={file_stats['unknown_label']}"
        )

        if max_items > 0 and stats["processed"] >= max_items:
            print(f"[hard_class] Reached max_items={max_items}. Stopping early.")
            break

    selected_by_class = {}
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

    balanced_selected: List[Dict[str, Any]] = []
    for cls_name in ("real", "pad", "deepfake", "makeup"):
        balanced_selected.extend(selected_by_class[cls_name])

    rng.shuffle(balanced_selected)
    split_index = math.floor(len(balanced_selected) * train_ratio)
    train_data = balanced_selected[:]  # keep current behavior
    eval_data = balanced_selected[split_index:]

    write_json(output_train, train_data)
    write_json(output_eval, eval_data)

    print("\n[hard_class] Summary")
    print(f"processed={stats['processed']} easy={stats['easy']} hard={stats['hard']}")
    print(
        f"hard_valid={stats['hard_valid']} forced_kept={stats['forced_kept']} "
        f"invalid_sample={stats['invalid_sample']} invalid_answers={stats['invalid_answers']}"
    )
    print(
        f"missing_image={stats['missing_image']} unknown_label={stats['unknown_label']} "
        f"file_error={stats['file_error']}"
    )

    for cls_name in ("real", "pad", "deepfake", "makeup"):
        print(
            f"{cls_name}: hard={hard_count_by_class[cls_name]} "
            f"forced={forced_count_by_class[cls_name]} "
            f"selected={len(selected_by_class[cls_name])}"
        )

    missing_forced = sorted(always_include_set - forced_files_found)
    print(f"always_include_files={sorted(always_include_set)}")
    print(f"always_include_files_found={sorted(forced_files_found)}")
    if missing_forced:
        print(f"always_include_files_missing={missing_forced}")

    for cls_name in ("real", "pad", "deepfake", "makeup"):
        overflow = len(forced_by_class[cls_name]) - sample_per_class
        if overflow > 0:
            print(
                f"warning: {cls_name} forced samples exceed target by {overflow}; "
                f"kept all forced samples."
            )

    stats["selected_total"] = len(balanced_selected)
    stats["train"] = len(train_data)
    stats["eval"] = len(eval_data)
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified MIDS merge/select script. Combines merge_mids_json.py and "
            "merge_check_mids_json.py behavior."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix",
        help="Directory containing split json files (for merge step).",
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default="/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/merged",
        help="Directory for merged json files and selection input.",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merge step and directly use --merged_dir.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hard_class", "legacy_name"],
        default="hard_class",
        help=(
            "Selection mode: "
            "hard_class = parse answers + get_label_all + class-balanced hard sampling; "
            "legacy_name = filename-based easy/hard sampling from old script."
        ),
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
        "--train_ratio",
        type=float,
        default=0.95,
        help="Split ratio used to create eval tail after shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffle.",
    )

    # hard_class mode
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=214386,
        help="hard_class mode: target hard samples per class (before forced overflow).",
    )
    parser.add_argument(
        "--always_include_files",
        nargs="*",
        default=["mids_dir_err3.json"],
        help=(
            "hard_class mode: filenames to include fully (hard samples) without random selection."
        ),
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="hard_class mode: debug cap on number of processed samples (0 means all).",
    )

    # legacy_name mode
    parser.add_argument(
        "--easy_n",
        type=int,
        default=200000,
        help="legacy_name mode: sampling cap for files containing 'easy'.",
    )
    parser.add_argument(
        "--hard_n",
        type=int,
        default=450000,
        help="legacy_name mode: sampling cap for files containing 'hard'.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_dir = Path(args.input_dir)
    merged_dir = Path(args.merged_dir)
    output_train = Path(args.output_train)
    output_eval = Path(args.output_eval)
    rng = random.Random(args.seed)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be between 0 and 1")
    if args.sample_per_class <= 0:
        raise ValueError("--sample_per_class must be > 0")
    if args.easy_n <= 0 or args.hard_n <= 0:
        raise ValueError("--easy_n and --hard_n must be > 0")

    if not args.skip_merge:
        print("[pipeline] Step 1/2: merge by prefix")
        merge_stats = merge_json_files_by_prefix(input_dir=input_dir, merged_dir=merged_dir)
        print(
            f"[pipeline] merge done: prefixes={merge_stats['prefixes']} files={merge_stats['files']} "
            f"items={merge_stats['merged_items']} file_errors={merge_stats['file_errors']}"
        )
    else:
        print("[pipeline] skip_merge=True; using existing merged_dir")

    print(f"[pipeline] Step 2/2: selection mode={args.mode}")
    if args.mode == "hard_class":
        sel_stats = select_hard_class_mode(
            merged_dir=merged_dir,
            output_train=output_train,
            output_eval=output_eval,
            sample_per_class=args.sample_per_class,
            always_include_files=args.always_include_files,
            train_ratio=args.train_ratio,
            rng=rng,
            max_items=args.max_items,
        )
    else:
        sel_stats = select_legacy_name_mode(
            merged_dir=merged_dir,
            output_train=output_train,
            output_eval=output_eval,
            easy_n=args.easy_n,
            hard_n=args.hard_n,
            train_ratio=args.train_ratio,
            rng=rng,
        )

    print(
        f"[pipeline] done: selected={sel_stats['selected_total']} "
        f"train={sel_stats['train']} eval={sel_stats['eval']}"
    )
    print(f"[pipeline] train_output={output_train}")
    print(f"[pipeline] eval_output={output_eval}")


if __name__ == "__main__":
    main()
