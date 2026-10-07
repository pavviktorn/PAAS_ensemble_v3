import os
from pathlib import Path
import random
import ntpath
from shutil import copyfile
from get_label import get_label_all

dataset_list = [
    # "PAD/dataset/from_nizar",
    # "from_nizar",
    # "from_nizar1",
    "/datasets/newout/PAD/dataset/Replay-Mobile",
]
root = Path(r"/datasets/newout")
n_toatl = 0
all_jpgs = []
sel_count = 280000
for dname in dataset_list:
    ds_root = os.path.join(root, dname)

    leaf_dirs = []

    # 1. Find leaf (end) subfolders
    for dirpath, dirnames, filenames in os.walk(ds_root):
        if not dirnames:  # no subdirectories -> it's an end-subfolder
            d = Path(dirpath)

            # all jpg files in this leaf folder
            jpgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
            jpgs = [
                str(p) for p in jpgs
                    if 0 == get_label_all(str(p))
                       and "Replay-Mobile" in str(p)
                    ]

            if not jpgs:
                continue

            all_jpgs.extend(jpgs)

k = min(sel_count, len(all_jpgs))
selected = random.sample(all_jpgs, k=k)
n_toatl += len(selected)
for pth in selected:
    head, fname = ntpath.split(pth)
    dst_path = r"/datasets/work/vLLM/data/sel_for_mids_4+13fmt/real"
    temp_path = head.replace('/datasets/newout', dst_path)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path, exist_ok=True)
    temp_path = os.path.join(temp_path, fname)
    copyfile(pth, temp_path)
    print(temp_path)

print(n_toatl)
