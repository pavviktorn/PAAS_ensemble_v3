import math
import os
import json
import random
from collections import defaultdict

# Directory containing the JSON files
INPUT_DIR = "/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix"     # change if needed
MERGED_DIR = "/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/merged"  # merged files written here
OUTPUT_FILE = "/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json"
EVAL_FILE = "/datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json"
EASY_N = 200000
HARD_N = 450000

# all = []
# with open("/datasets/newout/vqa_info/miss&makeup/eFFAA_dataset.json", "r", encoding="utf-8") as f:
#     data1 = json.load(f)
# with open("/datasets/newout/vqa_info/miss&makeup/eFFAA_misclssified_dataset.json", "r", encoding="utf-8") as f:
#     data2 = json.load(f)
# all.extend(data1)
# all.extend(data2)
# with open("/datasets/newout/vqa_info/miss&makeup/eFFAA_ext_missclassified.json", "w", encoding="utf-8") as out:
#     json.dump(all, out, indent=2, ensure_ascii=False)


def get_prefix(filename):
    """
    Extract prefix from filename.
    Example:
        mids_df_easy_1.json -> mids_df_easy
        mids_df_hard_7.json -> mids_df_hard
        mids_dir_err1_0.json -> mids_dir_err1
    """
    name = filename.replace(".json", "")
    parts = name.split("_")

    # If last part is a number, remove it
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    else:
        return name  # no numeric suffix → use entire name


def merge_json_files():
    if not os.path.exists(MERGED_DIR):
        os.makedirs(MERGED_DIR)

    groups = defaultdict(list)

    # Group files by prefix
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith(".json"):
            prefix = get_prefix(fname)
            groups[prefix].append(os.path.join(INPUT_DIR, fname))

    # Merge and output
    for prefix, files in groups.items():
        merged_data = []

        for f in sorted(files):  # sort for consistent ordering
            try:
                with open(f, "r") as infile:
                    data = json.load(infile)

                    # Add list or dict entries
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)

            except Exception as e:
                print(f"Error reading {f}: {e}")

        # Save merged file
        output_path = os.path.join(MERGED_DIR, f"{prefix}.json")
        with open(output_path, "w") as out:
            json.dump(merged_data, out, indent=2)

        print(f"Merged {len(files)} files → {output_path}")

def sample_from_list(data, target_n):
    """Return up to target_n random items from list data."""
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list to sample items.")
    if target_n >= len(data):
        return data  # take all if fewer than desired
    return random.sample(data, target_n)

def select_mids_trainset():
    all_selected = []

    for fname in os.listdir(MERGED_DIR):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(MERGED_DIR, fname)
        lower_name = fname.lower()

        # Decide how many items to take
        if "easy" in lower_name:
            target_n = EASY_N
        elif "hard" in lower_name:
            target_n = HARD_N
        else:
            target_n = None  # means take all

        if "real" in lower_name or "dir" in lower_name:
            target_n = None

        print(f"Processing: {fname}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"WARNING: {fname} root is not a list. Skipping this file.")
            continue

        if target_n is None:
            selected = data
            print(f"  Non-easy/non-hard file: taking all {len(selected)} items")
        else:
            selected = sample_from_list(data, target_n)
            print(f"  Taking {len(selected)} items (requested {target_n}, source size {len(data)})")

        all_selected.extend(selected)

    random.shuffle(all_selected)
    split_index = math.floor(len(all_selected) * 0.95)
    part1 = all_selected[:split_index]
    part2 = all_selected[split_index:]

    # Write final combined file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_selected, out, indent=2, ensure_ascii=False)

    # selected = sample_from_list(all_selected, HARD_N)
    with open(EVAL_FILE, "w", encoding="utf-8") as out:
        json.dump(part2, out, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Wrote {len(part1)} total items to {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_json_files()
    select_mids_trainset()
