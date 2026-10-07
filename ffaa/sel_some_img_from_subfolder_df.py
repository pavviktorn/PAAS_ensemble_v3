import os
from pathlib import Path
import random
import ntpath
from shutil import copyfile
from get_label import get_label_all

FAKE_LABEL = 2
dataset_list = [
    "from_nizar",
    "from_nizar1",
    "1m_faces_91__99",
    "add_free/70k-face-frames",
    "add_free/deepf_real_images",
    "add_free/DeepfDatabase",
    "add_free/deepfdatatrainvalid",
    "add_free/DeepFImages",
    "add_free/DFFMD",
    "add_free/RealvsFfaces",
    "add_free/styleGAN3_annotated",
    "AgedSyntheticImages",
    "CelebDF_v2",
    "DeeperForensics",
    "DEEPFAKE_CHALLENGE",
    "DeepFakeFace",
    "DeepfakeTIMIT",
    "DFFD",
    "DFGC-2021",
    # "DFGC-2022",
    "DFMNIST+",
    "FF++_HifiFace",
    "how_fmc",
    "iFakeFaceDB",
    "MegaFS",
    "new_df",
]
root = Path(r"/datasets/newout")
n_toatl = 0
for dname in dataset_list:
    ds_root = os.path.join(root, dname)
    subfolder_count = 0
    for _, dirs, _ in os.walk(ds_root):
        subfolder_count += len(dirs)

    is_percent = False
    sel_percent = 0.05
    sel_count = 1
    if subfolder_count < 1000:
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
                    if get_label_all(str(p)) == FAKE_LABEL
                    and "/real/" not in str(p)
                    and "_mask" not in str(p)
                    and "_depth" not in str(p)
                    and "-real" not in str(p)
                    and "original_" not in str(p)
                    and "Resource" not in str(p)
                    ]

            if not jpgs:
                continue

            if is_percent:
                k = min(len(jpgs), int(len(jpgs) * sel_percent))
            else:
                k = min(sel_count, len(jpgs))
            selected = random.sample(jpgs, k=k)
            n_toatl += len(selected)
            for pth in selected:
                head, fname = ntpath.split(pth)
                dst_path = r"/datasets/work/vLLM/data/sel_for_mids_4+13fmt/fake/deepfake"
                temp_path = head.replace('/datasets/newout', dst_path)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path, exist_ok=True)
                    print(temp_path)
                temp_path = os.path.join(temp_path, fname)
                copyfile(pth, temp_path)

print(n_toatl)
