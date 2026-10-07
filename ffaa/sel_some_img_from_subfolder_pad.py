import os
from pathlib import Path
import random
import ntpath
from shutil import copyfile
from get_label import get_label_all

FAKE_LABEL = 1

dataset_list = [
    "from_nizar",
    "CeFA",
    "datatang_sample",
    "Oulu_NPU",
    "Replay-Attack",
    "Replay-Mobile",
    "Rose",
    "SiW",
    "SiW-Mv2",
]
root = Path(r"/datasets/newout/PAD/dataset")

def proc_dataset(dname, get_label, sel_count = 1, sel_percent = 0.05):
    n_sel = 0
    ds_root = os.path.join(root, dname)
    subfolder_count = 0
    for _, dirs, _ in os.walk(ds_root):
        subfolder_count += len(dirs)

    is_percent = False
    # sel_percent = 0.05
    # sel_count = 1
    if subfolder_count > 1000:
        is_percent = True

    leaf_dirs = []

    # 1. Find leaf (end) subfolders
    for dirpath, dirnames, filenames in os.walk(ds_root):
        if not dirnames:  # no subdirectories -> it's an end-subfolder
            d = Path(dirpath)

            # all jpg files in this leaf folder
            jpgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
            jpgs = [
                p for p in jpgs
                    if get_label(str(p)) == FAKE_LABEL
                    and "_mask" not in str(p)
                    and "_depth" not in str(p)
                    ]

            if not jpgs:
                continue

            if is_percent:
                k = min(len(jpgs), int(len(jpgs) * sel_percent))
            else:
                k = min(sel_count, len(jpgs))
            selected = random.sample(jpgs, k=k)
            n_sel += len(selected)
            for pth in selected:
                head, fname = ntpath.split(pth)
                dst_path = r"/datasets/work/vLLM/data/sel_for_mids_4+13fmt/fake/pad"
                temp_path = head.replace('/datasets/newout', dst_path)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path, exist_ok=True)
                    print(temp_path)
                temp_path = os.path.join(temp_path, fname)
                copyfile(pth, temp_path)

    return n_sel

n_toatl = 0

n_toatl += proc_dataset('from_nizar', get_label_all, sel_count=10)
n_toatl += proc_dataset('CeFA', get_label_all)
n_toatl += proc_dataset('datatang_sample', get_label_all)
n_toatl += proc_dataset('Oulu_NPU', get_label_all, sel_count=5)
n_toatl += proc_dataset('Replay-Attack', get_label_all, sel_count=5)
n_toatl += proc_dataset('Replay-Mobile', get_label_all, sel_count=6)
n_toatl += proc_dataset('Rose', get_label_all, sel_count=5)
n_toatl += proc_dataset('SiW', get_label_all, sel_count=2)
n_toatl += proc_dataset('SiW-Mv2', get_label_all, sel_count=10)

print(n_toatl)
