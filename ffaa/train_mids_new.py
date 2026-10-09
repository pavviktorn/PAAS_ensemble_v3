import random

import torch
from dataclasses import dataclass, field
from PIL import Image
import transformers
from transformers import T5Tokenizer, CLIPProcessor
from torch.utils.data import Dataset
import deepspeed
from typing import Optional
from albumentations import Compose, FancyPCA, HueSaturationValue, ImageOnlyTransform, OneOf, ImageCompression, GaussNoise, GaussianBlur, MotionBlur
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import datetime
import os
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from utils.mids_utils import flatten_tuple_list
from mids.selector import make_decision_batch

def rank0_print(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)

@dataclass
class ModelArguments:
    hidden_dim: int = 768
    version: str = 'v1'
    image_model_path: str = 'models/clip-vit-large-patch14-336'
    text_model_path: str = 'models/t5-base'
    init_model_path: str = './checkpoints_4+13fmt/mids.pth'


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = 24
    per_device_val_batch_size: int = 12
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 1e-5
    num_train_epochs: int = 2
    unfreeze_vision_encoder_last_layers: int = 2
    output_dir: Optional[str] = None
    lambda_rank: float = 0.5
    rank_margin: float = 0.1
    eval_every_steps: int = 5000          # evaluate on the eval set every N optimizer steps
    select_metric: str = 'acc'            # acc | auc | ap  -> which eval metric picks best.pth

class RandomSizedCenterCrop(ImageOnlyTransform):
    def __init__(self, min_scale=0.8, max_scale=1.0, cx=0.5, cy=0.4, p=0.5):
        super().__init__(p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.cx = cx
        self.cy = cy

    def apply(self, image, **params):
        h, w = image.shape[:2]

        scale = random.uniform(self.min_scale, self.max_scale)
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        y1 = int(self.cy * h - crop_h / 2)
        x1 = int(self.cx * w - crop_w / 2)

        y1 = max(0, min(y1, h - crop_h))
        x1 = max(0, min(x1, w - crop_w))

        return image[y1:y1+crop_h, x1:x1+crop_w]
    
class MLLMAnswersDataset(Dataset):
    def __init__(self, data_path, processor, mode='train'):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.mode = mode
        self.data_dir = os.path.dirname(data_path)
        self.clip_processor = processor
        self.aug = Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            RandomSizedCenterCrop(min_scale=0.9, max_scale=1.0, p=0.5),
            GaussNoise(p=0.2),
            MotionBlur(p=0.2),
            GaussianBlur(blur_limit=3, p=0.2),
            OneOf([FancyPCA(), HueSaturationValue()], p=0.7),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # # BGR → RGB (critical) github code bug 20260305
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #zzzzzz this is critical for CLIP, otherwise the performance will be very bad. But for some reason, it seems that the original code in sel_hard_samples_from_folder_batch.py does not have this step, which may cause some performance drop. So I add this step here for compatibility with train_mids.py and better performance.
        if self.mode == 'train':
            image = self.aug(image=image)["image"]
        image = Image.fromarray(image)
        image = self.clip_processor(images=image, return_tensors="pt")['pixel_values']

        cls_label = item['cls_label']
        answers = item['answers']
        texts = [answer['content'] for answer in answers]
        answers_result = [answer['result'] for answer in answers[:3]]
        labels = [answer['label'] for answer in answers]

        return {
            'image': image,
            'cls_label': cls_label,
            'texts': texts,
            'answers_result': answers_result,
            'labels': labels,
        }

def load_mids(model_path, model):

    model_state_dict = model.state_dict()

    finetuned_state_dict = torch.load(model_path)
    finetuned_state_dict = {key.replace('module.', ''): value for key, value in finetuned_state_dict.items()}

    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    return model

def val_model(tag, model, tokenizer, device, val_dataloader):
    model.eval()

    y_trues = []
    y_preds = []
    y_scores = []

    with tqdm(val_dataloader, desc=f"Val[{tag}]", disable=(torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)) as t:
        for batch in t:
            batch_size = len(batch['image'])
            images = batch['image'].squeeze(1).to(device)
            cls_labels = batch['cls_label'].to(device)
            flattened_texts = flatten_tuple_list(batch['texts'])
            answers_result = flatten_tuple_list(batch['answers_result'])
            input_ids = tokenizer(flattened_texts, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).to(device)

            logits = model(input_ids, images, None, batch_size, 1, 1)['logits']
            scores = F.softmax(logits, dim=2)
            best_answer_idxs, preds, match_scores, forgery_scores = make_decision_batch(answers_result, scores)

            y_trues.extend(cls_labels.detach().cpu().numpy().tolist())
            y_preds.extend(preds)
            y_scores.extend(forgery_scores)

    # gather the sharded predictions across DDP ranks so the metric is on the FULL eval set and is
    # identical on every rank (keeps the best.pth decision consistent under DistributedSampler).
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        bucket = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(bucket, (y_trues, y_preds, y_scores))
        y_trues = [v for part in bucket for v in part[0]]
        y_preds = [v for part in bucket for v in part[1]]
        y_scores = [v for part in bucket for v in part[2]]

    acc = accuracy_score(y_trues, y_preds)
    try:
        auc = roc_auc_score(y_trues, y_scores)
        ap = average_precision_score(y_trues, y_scores)
    except ValueError:                     # e.g. only one class present in a small eval slice
        auc = ap = float('nan')

    return acc, auc, ap


def compute_match_score_from_logits(logits, answers_result, eps=1e-8):
    """
    logits: [B, K, 4]
    answers_result: list length B*K flattened OR nested list [B][K]
    returns: S_m tensor [B, K]
    """
    probs = F.softmax(logits, dim=2)
    m0 = probs[:, :, 0]
    m1 = probs[:, :, 1]
    m2 = probs[:, :, 2]
    m3 = probs[:, :, 3]

    batch_size, k_answers, _ = logits.shape
    if answers_result and isinstance(answers_result[0], (list, tuple)):
        answers_nested = answers_result
    else:
        answers_nested = [
            answers_result[i * k_answers:(i + 1) * k_answers]
            for i in range(batch_size)
        ]

    ans_is_fake = torch.tensor(
        [[1 if r == 'fake' else 0 for r in answers] for answers in answers_nested],
        device=logits.device,
    ).bool()

    sm_real = m0 / (m0 + m2 + eps)
    sm_fake = m3 / (m3 + m1 + eps)
    return torch.where(ans_is_fake, sm_fake, sm_real)


def ranking_loss(S, cls_labels, answers_result, margin=0.1):
    """
    S: [B, K] match scores
    cls_labels: [B]
    answers_result: nested list [B][K]
    """
    batch_size, _ = S.shape
    loss = S.new_tensor(0.0)

    for i in range(batch_size):
        gt_is_fake = bool(cls_labels[i].item())

        is_correct = torch.tensor(
            [(r == 'fake') == gt_is_fake for r in answers_result[i]],
            device=S.device,
            dtype=torch.bool,
        )

        if is_correct.any() and (~is_correct).any():
            s_pos = S[i][is_correct].max()
            s_neg = S[i][~is_correct].max()
            # Smooth logistic ranking: log(1 + exp(-(s_pos - s_neg)))
            loss += F.softplus(-(s_pos - s_neg))

    return loss / batch_size

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    deepspeed.init_distributed()
    world_size = torch.distributed.get_world_size()

    # clip processor
    clip_processor = CLIPProcessor.from_pretrained(model_args.image_model_path)
    # dataset
    training_dataset = MLLMAnswersDataset(data_args.data_path, clip_processor, 'train')
    val_dataset = MLLMAnswersDataset(data_args.val_data_path, clip_processor, 'val')
    sampler = DistributedSampler(training_dataset)
    val_sampler = DistributedSampler(val_dataset)
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=16,
        sampler=sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_args.per_device_val_batch_size,
        num_workers=16,
        sampler=val_sampler,
    )

    # create model
    tokenizer = T5Tokenizer.from_pretrained(model_args.text_model_path, use_fast=False, legacy=False)

    model_kwargs = {
        # 'loss_type': training_args.loss_type,
        # 'ratio': training_args.ratio,
        # 'temperature': training_args.temperature,
        # 'num_classes': model_args.num_classes,
        'image_model_path': model_args.image_model_path,
        'text_model_path': model_args.text_model_path
    }

    if model_args.version == 'v1':
        from mids.mids_arch import MIDS
        model = MIDS(model_args.hidden_dim, **model_kwargs)  
    else:
        NotImplementedError

    # model = load_mids('./checkpoints/ffaa-mistral-7b/mids.pth', model)  # zzzzzz
    model = load_mids(model_args.init_model_path, model)  # zzzzzz

    model.freeze_text_encoder_parameters()
    model.freeze_image_encoder_parameters()

    if training_args.unfreeze_vision_encoder_last_layers > 0:
        model.unfreeze_vision_encoder(training_args.unfreeze_vision_encoder_last_layers)
    
    
    def init_non_frozen_params(model):
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear)) or not module.weight.requires_grad:
                continue
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    # model.apply(init_non_frozen_params) #zzzzz
    rank0_print("Trainable Parameters:")
    finetuned_layers = ['module.' + name for name, param in model.named_parameters() if param.requires_grad]
    rank0_print(finetuned_layers)
    
    # optim
    optimizer_grouped_parameters = [
        {"params": [p for p in model.parameters() if p.requires_grad]}
    ]   
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    
    # get some common args here
    num_epochs = training_args.num_train_epochs
    output_dir = training_args.output_dir

    # deepspeed config
    ds_config = {
        "train_batch_size": training_args.per_device_train_batch_size * world_size,  # 所有 GPU 上的总批量大小
        "fp16": {
            "enabled": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.learning_rate,
                "warmup_num_steps": int(training_args.warmup_ratio * num_epochs * len(training_dataloader))
            }
        },
        "steps_per_print": 200000
    }

    # init DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # checkpoint save dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f'mids_{model_args.version}_{timestamp}')
    if training_args.unfreeze_vision_encoder_last_layers <= 0:
        save_dir = save_dir+'_freeze'

    rank0_print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_metric = -1.0
    best_step = 0
    global_step = 0
    eval_every = max(1, int(training_args.eval_every_steps))
    metric_name = training_args.select_metric if training_args.select_metric in ("acc", "auc", "ap") else "acc"

    def run_eval_and_save(tag):
        """Evaluate on the eval set; overwrite best.pth when the selection metric improves."""
        nonlocal best_metric, best_step
        val_acc, val_auc, val_ap = val_model(tag, model, tokenizer, device, val_dataloader)
        metric = {"acc": val_acc, "auc": val_auc, "ap": val_ap}[metric_name]
        rank0_print(f"[eval @ {tag}] step={global_step} ACC={val_acc:.4f} AUC={val_auc:.4f} "
                    f"AP={val_ap:.4f} | select {metric_name}={metric:.4f} (best {best_metric:.4f} @ step {best_step})")
        if metric >= best_metric:
            best_metric, best_step = metric, global_step
            if torch.distributed.get_rank() == 0:
                sd = {name: param for name, param in model.state_dict().items() if name in finetuned_layers}
                torch.save(sd, os.path.join(save_dir, 'best.pth'))
                rank0_print(f"  -> new best {metric_name}={metric:.4f} @ step {global_step}; saved best.pth")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        y_trues = []
        y_preds = []
        y_scores = []

        sampler.set_epoch(epoch)

        # train
        model.train()
        with tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for batch in t:
                optimizer.zero_grad()
                batch_size = len(batch['image'])

                images = batch['image'].squeeze(1).to(device)
                cls_labels = batch['cls_label'].to(device)
                flattened_texts = flatten_tuple_list(batch['texts'])
                answers_result = flatten_tuple_list(batch['answers_result'])
                
                labels = torch.tensor(flatten_tuple_list(batch['labels'])).to(device) #zzzzzz
                # labels = torch.tensor(batch['labels'], dtype=torch.long).view(-1).to(device)
                
                input_ids = tokenizer(flattened_texts, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True).to(device)
                output = model(input_ids, images, labels, batch_size, 1, 1)

                loss_ce = output['cls_loss']
                logits = output['logits'][:, :3, :]

                k_answers = logits.size(1)
                answers_nested = [
                    answers_result[i * k_answers:(i + 1) * k_answers]
                    for i in range(batch_size)
                ]

                S = compute_match_score_from_logits(logits, answers_nested)
                loss_rank = training_args.lambda_rank * ranking_loss(
                    S,
                    cls_labels,
                    answers_nested,
                    margin=training_args.rank_margin,
                )

                loss = loss_ce + loss_rank

                model.backward(loss)
                model.step()
                global_step += 1

                epoch_loss += loss.item()

                scores = F.softmax(logits, dim=2)
                best_answer_idxs, preds, match_scores, forgery_scores = make_decision_batch(answers_result, scores[:,:3,:])

                y_trues.extend(cls_labels.detach().cpu().numpy().tolist())
                y_preds.extend(preds)
                y_scores.extend(forgery_scores)

                train_acc = accuracy_score(y_trues, y_preds)
                train_auc = roc_auc_score(y_trues, y_scores)
                train_ap = average_precision_score(y_trues, y_scores)

                t.set_postfix({'loss': loss.item(), 'cls': loss_ce.item(), 'rank': loss_rank.item(), 'acc': train_acc, 'auc': train_auc, 'ap': train_ap})
                # rank0_print({'loss': loss.item(), 'acc': train_acc, 'auc': train_auc, 'ap': train_ap})

                # ---- evaluate on the eval set every `eval_every_steps` steps; keep best.pth ----
                if global_step % eval_every == 0:
                    run_eval_and_save(f"step{global_step}")

        epoch_loss /= len(training_dataloader)
        rank0_print(f"Epoch {epoch+1}/{num_epochs} finished - train Loss: {epoch_loss:.4f}")

        # end of epoch: evaluate on the eval set (prints ACC/AUC/AP, updates best.pth) and
        # snapshot this epoch's weights as <epoch>.pth (1.pth, 2.pth, ...)
        run_eval_and_save(f"epoch{epoch+1}")
        if torch.distributed.get_rank() == 0:
            sd = {name: param for name, param in model.state_dict().items() if name in finetuned_layers}
            torch.save(sd, os.path.join(save_dir, f'{epoch+1}.pth'))
            rank0_print(f"  -> saved epoch checkpoint {os.path.join(save_dir, f'{epoch+1}.pth')}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # final evaluation (also updates best.pth if the last state is best) + a final.pth snapshot
    run_eval_and_save("final")
    if torch.distributed.get_rank() == 0:
        finetuned_state_dict = {name: param for name, param in model.state_dict().items() if name in finetuned_layers}
        torch.save(finetuned_state_dict, os.path.join(save_dir, f'final.pth'))

    rank0_print(f"Best {metric_name}={best_metric:.4f} achieved at step {best_step}; best.pth in {save_dir}")

if __name__ == "__main__":
    train()
