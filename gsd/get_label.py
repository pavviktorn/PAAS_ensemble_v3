import os
from pathlib import Path
import random
import ntpath
from shutil import copyfile

REAL = 0
PAD = 1
DEEPFAKE = 2
MAKEUP = 3
UNKNOWN = -1

def get_label_normal(src_path):
    if '/fake_excluded/' in src_path or 'makeup' in src_path.lower():
        label = MAKEUP
    elif '/fake' in src_path and '/pad/' in src_path.lower():
        label = PAD
    elif '/fake' in src_path and '/deepfake/' in src_path.lower():
        label = DEEPFAKE
    elif '/real' in src_path:
        label = REAL
    else:
        label = UNKNOWN

    return label

def get_label_datatang(src_path):
    head, fname = ntpath.split(src_path)
    tag = str(src_path).split('/')[-2]
    if 'original' in fname:
        label = REAL
    elif 'anti_' in fname:
        label = PAD
    elif 'anti_' in tag:
        label = PAD
    else:
        label = REAL

    return label

def get_label_3DMAD(src_path):
    head, fname = ntpath.split(src_path)
    if '_' in fname:
        category = int(fname.split('_')[1])
    else:
        return UNKNOWN
    if category == 3:
        label = PAD
    else:
        label = REAL

    return label

def get_label_CelebA(src_path):
    if '/spoof/' in src_path:
        label = PAD
    elif '/live/' in src_path:
        label = REAL
    else:
        label = UNKNOWN

    return label

def get_label_ERPA(src_path):
    if '/no_mask/' in src_path:
        label = REAL
    elif '/png_ir/' in src_path or '/myvideoIR/' in src_path:
        label = UNKNOWN
    else:
        label = PAD

    return label

def get_label_Rose(src_path):
    fname = str(src_path).split('/')[-2]
    if '_' in fname:
        tag = fname.split('_')[0]
    else:
        return UNKNOWN
    if tag == 'G':
        label = REAL
    else:
        label = PAD

    return label

def get_label_SWAX(src_path):
    if '/Dummies/' in src_path:
        label = PAD
    else:
        label = REAL

    return label

def get_label_CeFA(src_path):
    return PAD

def get_label_Oulu_NPU(src_path):
    tag = str(src_path).split('/')[-2]
    tag = tag.split('_')[-1]
    if tag == '1':
        label = REAL # real
    else:
        label = PAD

    return label

def get_mydata(src_path):
    if '/fake/' in src_path and '/pad/' in src_path.lower():
        label = PAD
    elif '/real/' in src_path and 't4liveness' in src_path.lower():
        label = UNKNOWN
    elif '/real/' in src_path:
        label = REAL
    else:
        label = UNKNOWN

    return label

def get_mywebcam(src_path):
    if '/fake/' in src_path and '/pad/' in src_path.lower():
        label = PAD
    else:
        label = UNKNOWN

    return label

HiFiMask_dic = {}
try:
    f = open("/datasets/datasets/PAD/dataset/HiFiMask/all_label.txt", 'r')
    lines = f.readlines()
    f.close()
except OSError:  # dataset label file only exists on the data-prep server; absent -> UNKNOWN below
    lines = []
for i in range(len(lines)):
    line = lines[i]
    elm = line.split(' ')
    src_path = elm[0]
    src_path = src_path.replace("png", "jpg")
    if (int(elm[1]) == 0):
        label = PAD
    elif (int(elm[1]) == 1):
        label = REAL #real
    else:
        continue
    HiFiMask_dic[src_path] = label
def get_label_HiFiMask(src_path):
    key = "/".join(src_path.split("/")[-3:])
    if key in HiFiMask_dic:
        label = HiFiMask_dic.get(key)
        return label
    else:
        return UNKNOWN

SuHiFiMask_dic = {}
try:
    f = open("/datasets/datasets/PAD/dataset/SuHiFiMask/all_label.txt", 'r')
    lines = f.readlines()
    f.close()
except OSError:  # dataset label file only exists on the data-prep server; absent -> UNKNOWN below
    lines = []
for i in range(len(lines)):
    line = lines[i]
    elm = line.split(' ')
    src_path = elm[0]
    src_path = src_path.replace("png", "jpg")
    if (int(elm[1]) == 0):
        label = PAD
    elif (int(elm[1]) == 1):
        label = REAL #real
    else:
        continue
    SuHiFiMask_dic[src_path] = label
def get_label_SuHiFiMask(src_path):
    key = "/".join(src_path.split("/")[-4:])
    if key in HiFiMask_dic:
        label = SuHiFiMask_dic.get(key)
        return label
    else:
        return UNKNOWN

def get_label_all(src_path):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    ext = Path(src_path).suffix.lower()
    if not ext in IMAGE_EXTS:
        return UNKNOWN
    head, fname = ntpath.split(src_path)
    if "_mask" in fname or "_depth" in fname:
        return UNKNOWN
    
    if "/pad/" in src_path.lower():
        if "3DMAD" in src_path:
            return get_label_3DMAD(src_path)
        elif "CeFA" in src_path:
            return get_label_CeFA(src_path)
        elif "CelebA-Spoof" in src_path:
            return get_label_CelebA(src_path)
        elif "datatang_sample" in src_path:
            return get_label_datatang(src_path)
        elif "mydata" in src_path:
            return get_mydata(src_path)
        elif "mywebcam" in src_path:
            return get_mywebcam(src_path)
        elif "Oulu_NPU" in src_path:
            return get_label_Oulu_NPU(src_path)
        elif "Rose" in src_path:
            return get_label_Rose(src_path)
        elif ("SiW-Mv2" in src_path
              or "CVPR2023-ASF" in src_path
              or "CSMAD" in src_path
              or "from_nizar" in src_path
              or "Replay-Mobile" in src_path
              or "Replay-Attack" in src_path):
            return get_label_normal(src_path)
        elif "SiW" in src_path:
            return get_label_CelebA(src_path)
        elif "SWAX" in src_path:
            return get_label_SWAX(src_path)
        elif "ERPA" in src_path:
            return get_label_ERPA(src_path)
        elif "HiFiMask" in src_path:
            return get_label_HiFiMask(src_path)
        elif "SuHiFiMask" in src_path:
            return get_label_SuHiFiMask(src_path)
        else:
            return get_label_normal(src_path)
    elif "/makeup/" in src_path.lower():
        return MAKEUP
    elif ("DFDC" in src_path
          or "DFGC-2022" in src_path):
        return UNKNOWN
    else: # deepfake
        if "FF++_HifiFace" in src_path:
            return DEEPFAKE  # deepfake
        elif ("/real/" in src_path
                or "/real " in src_path
                or "-real" in src_path
                or "original_" in src_path
                or "Resource" in src_path):
            return REAL # real
        elif (
            "/fake/" in src_path
            or "/fake " in src_path
            or "/fake_" in src_path
            or "/deepfake/" in src_path
            or "-synthesis" in src_path
            or "manipulated_" in src_path
        ):
            return DEEPFAKE # deepfake
        else:
            return UNKNOWN
