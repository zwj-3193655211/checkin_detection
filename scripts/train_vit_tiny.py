"""
Phase 3: 自训 ViT-Tiny 端到端（晨读/晨跑二分类 + 11 维特征预测），三支决策评估
================================================================================

目标：训练一个专精本项目、且不裁剪/不变形图片的模型，与 CLIP / SigLIP 管线公平对比。

关键设计：
- 骨干: timm vit_tiny_patch16_224（ImageNet augreg 预训练，本地权重
        data/models/vit_tiny/vit_tiny.safetensors）。
- 输入: AR 保持 letterbox 到 224x224（缩放短边适配 + 居中补边），
        零裁剪、零拉伸 —— 直接解决 CLIP CenterCrop 丢失 ~35% 像素的问题。
- 双头: 分类头(2 类) + 特征头(11 维 sigmoid)。
- 三支决策: 复用与 CLIP/SigLIP 管线完全一致的阈值（src/config），保证可比性。
- 多 seed: --all-seeds 训练 42/123/777，每个 seed 保存 checkpoint 与原始预测，
          供 Phase 4 统一做逐特征阈值调优与对比矩阵。

用法（从项目根目录）：
    python scripts/train_vit_tiny.py --all-seeds
    python scripts/train_vit_tiny.py --seed 42
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSIFIER_TEMPERATURE, FEATURE_TEMPERATURE,
    ALPHA_AUTO_PASS, ALPHA_REVIEW, MIN_FEATURES,
    RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
WEIGHTS = PROJECT_ROOT / "data" / "models" / "vit_tiny" / "vit_tiny.safetensors"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FEATURE_INDEX = {
    "人脸": 0, "蓝色桌子": 1, "教室": 2, "投影幕布": 3,
    "跑道": 4, "天空": 5, "绿地": 6, "树木": 7,
    "旗杆": 8, "号码布": 9, "主席台": 10,
}
CLASS_FEATURES = {
    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台'],
}


def letterbox(img):
    """AR 保持缩放 + 居中补边到 IMG_SIZE 正方形。零裁剪、零拉伸。"""
    w, h = img.size
    scale = IMG_SIZE / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    canvas.paste(img, ((IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2))
    return canvas


def build_feature_label(label_info):
    """与 train_mlp.prepare_data 完全一致的 11 维标签构造（含无标注回退）。"""
    fl = [0] * 11
    saved = label_info.get('features', {})
    for k, idx in FEATURE_INDEX.items():
        if saved.get(k, False):
            fl[idx] = 1
    if sum(fl) == 0:
        if saved.get('人脸', False):
            fl[0] = 1
        name = label_info.get('label', '')
        if name == '晨读':
            for k, idx in [('蓝色桌子', 1), ('教室', 2), ('投影幕布', 3)]:
                if saved.get(k, False):
                    fl[idx] = 1
        elif name == '晨跑':
            for k, idx in [('跑道', 4), ('天空', 5), ('绿地', 6), ('树木', 7),
                           ('旗杆', 8), ('号码布', 9), ('主席台', 10)]:
                if saved.get(k, False):
                    fl[idx] = 1
    return fl


def main_label_of(name):
    if name == '晨读':
        return 0
    if name == '晨跑':
        return 1
    return -1


class CheckinDataset(Dataset):
    def __init__(self, file_list, labels, is_train=False, augment=False):
        self.items = []
        for fn in file_list:
            li = labels.get(fn, {})
            name = li.get('label', '未知')
            if is_train and name not in ['晨读', '晨跑']:
                continue
            self.items.append((fn, main_label_of(name), build_feature_label(li)))
        self.augment = augment
        if augment:
            from torchvision.transforms import RandAugment
            self.raug = RandAugment(num_ops=2, magnitude=9)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn, ml, fl = self.items[i]
        img = Image.open(RAW_DIR / fn).convert("RGB")
        img = letterbox(img)
        if self.augment:
            img = self.raug(img)
        t = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 255.0
        t = (t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return t, ml, torch.tensor(fl, dtype=torch.float32)


class ViTCheckin(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # (B, 192)
        self.cls_head = nn.Linear(192, 2)
        nn.init.zeros_(self.cls_head.bias)
        self.feat_head = nn.Linear(192, 11)

    def forward(self, x):
        f = self.backbone(x)
        return self.cls_head(f), self.feat_head(f)


def load_backbone():
    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
    sd = load_file(str(WEIGHTS))
    sd = {k.replace('model.', ''): v for k, v in sd.items()}
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    print(f"[backbone] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    return backbone


def three_way(prob_conf, prob_feat, y_main, feat_thresh):
    """prob_conf: (N,) softmax 最大置信度；prob_feat: (N,11) sigmoid 概率；
    feat_thresh: (11,) 逐特征阈值。返回三支决策指标。"""
    miss = review = 0
    for i in range(len(y_main)):
        pl = '晨读' if prob_conf[i].argmax().item() == 0 else '晨跑'
        c = prob_conf[i].max().item()
        tl = int(y_main[i])
        cf = CLASS_FEATURES[pl]
        matched = sum(1 for f, fi in FEATURE_INDEX.items()
                      if prob_feat[i, fi].item() > feat_thresh[fi].item() and f in cf)
        r1 = pl == '晨跑' and c >= ALPHA_AUTO_PASS and matched >= RUN_FEATURE_THRESH
        r2 = pl == '晨读' and c >= ALPHA_AUTO_PASS and matched >= READ_FEATURE_THRESH
        r3 = matched < MIN_FEATURES
        r4 = c < ALPHA_REVIEW
        decision = 0
        if r1 or r2:
            decision = 0
        elif r3 or r4:
            decision = 1
        if tl == -1 and decision == 0:
            miss += 1
        if decision == 1:
            review += 1
    n = len(y_main)
    anom = sum(1 for v in y_main if v == -1)
    return dict(miss_rate=miss / anom * 100 if anom else 0.0,
                review_rate=review / n * 100,
                miss_count=miss, review_count=review,
                anomaly_total=anom, test_size=n)


def default_feat_thresh():
    return torch.tensor([0.66 if i < 4 else 0.60 for i in range(11)])


def train_one(seed, epochs=40, batch=48, lr=3e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[seed {seed}] device={device}")

    labels = json.load(open(PROJECT_ROOT / 'data' / 'labels.json', encoding='utf-8'))
    labels = labels.get('labels', labels)
    split = json.load(open(PROJECT_ROOT / 'data' / 'split_config.json', encoding='utf-8'))

    train_ds = CheckinDataset(split['train_files'], labels, is_train=True, augment=True)
    val_ds = CheckinDataset(split['val_files'], labels, is_train=False, augment=False)
    test_ds = CheckinDataset(split['test_files'], labels, is_train=False, augment=False)
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)

    backbone = load_backbone()
    model = ViTCheckin(backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cls_crit = nn.CrossEntropyLoss()
    feat_crit = nn.BCEWithLogitsLoss()

    ckpt = PROJECT_ROOT / 'data' / f'vit_tiny_seed{seed}.pt'
    best_score = -1e9
    best_vm = None
    ft = default_feat_thresh()

    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        n = 0
        for x, ml, fl in train_dl:
            x, ml, fl = x.to(device), ml.to(device), fl.to(device)
            cl, fh = model(x)
            normal = ml >= 0
            loss = cls_crit(cl[normal], ml[normal]) + 0.5 * feat_crit(fh, fl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
            n += len(x)
        sched.step()

        model.eval()
        with torch.no_grad():
            cvals, fvals, yv = [], [], []
            for x, ml, fl in val_dl:
                x = x.to(device)
                cl, fh = model(x)
                cvals.append(torch.softmax(cl.cpu() / CLASSIFIER_TEMPERATURE, dim=1))
                fvals.append(torch.sigmoid(fh.cpu() / FEATURE_TEMPERATURE))
                yv.append(ml)
            cvals = torch.cat(cvals)
            fvals = torch.cat(fvals)
            yv = torch.cat(yv).tolist()
        vm = three_way(cvals, fvals, yv, ft)
        score = (1000.0 if vm['miss_rate'] == 0 else 0.0) - vm['review_rate']
        if score > best_score:
            best_score = score
            best_vm = dict(vm)
            torch.save(model.state_dict(), ckpt)
        print(f"[seed {seed} ep{ep}] loss={tl/n:.4f} val miss={vm['miss_rate']:.2f}% "
              f"review={vm['review_rate']:.2f}% best={best_score:.1f}")

    # ---- 最佳 checkpoint 上做测试评估 + 保存原始预测 ----
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        cts, fts, yt, cvs, fvs, yv = [], [], [], [], [], []
        for x, ml, fl in test_dl:
            x = x.to(device)
            cl, fh = model(x)
            cts.append(torch.softmax(cl.cpu() / CLASSIFIER_TEMPERATURE, dim=1))
            fts.append(torch.sigmoid(fh.cpu() / FEATURE_TEMPERATURE))
            yt.append(ml)
        for x, ml, fl in val_dl:
            x = x.to(device)
            cl, fh = model(x)
            cvs.append(torch.softmax(cl.cpu() / CLASSIFIER_TEMPERATURE, dim=1))
            fvs.append(torch.sigmoid(fh.cpu() / FEATURE_TEMPERATURE))
            yv.append(ml)
    cts, fts, yt = torch.cat(cts), torch.cat(fts), torch.cat(yt).tolist()
    cvs, fvs, yv = torch.cat(cvs), torch.cat(fvs), torch.cat(yv).tolist()

    tm_ = three_way(cts, fts, yt, ft)
    pred_norm = cts.argmax(1)
    yn = torch.tensor(yt)
    norm_mask = yn >= 0
    acc = (pred_norm[norm_mask] == yn[norm_mask]).float().mean().item() * 100

    # 保存原始预测，供 Phase 4 统一做逐特征阈值调优
    pred_path = PROJECT_ROOT / 'data' / f'vit_tiny_preds_seed{seed}.pt'
    torch.save({
        'val_cls': cvs, 'val_feat': fvs, 'val_y': yv,
        'test_cls': cts, 'test_feat': fts, 'test_y': yt,
    }, pred_path)

    res = dict(seed=seed, test=tm_, head_acc=acc, val_best=best_vm,
               ckpt=str(ckpt), preds=str(pred_path))
    out = PROJECT_ROOT / 'data' / f'vit_tiny_results_seed{seed}.json'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=2)
    print(f"[seed {seed}] TEST miss={tm_['miss_rate']:.2f}% review={tm_['review_rate']:.2f}% "
          f"head_acc={acc:.2f}% | preds saved -> {pred_path.name}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--all-seeds', action='store_true')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    if a.all_seeds:
        # 断点续跑：跳过已经产出 results JSON 的 seed（checkpoint/预测齐全即视为完成）
        pending = []
        for s in [42, 123, 777]:
            res_path = PROJECT_ROOT / 'data' / f'vit_tiny_results_seed{s}.json'
            if res_path.exists():
                print(f"[skip] seed {s} 已完成（{res_path.name} 存在），跳过。")
            else:
                pending.append(s)
        if not pending:
            print("[all-seeds] 所有 seed 均已完成，无需训练。")
            return
        print(f"[all-seeds] 待训练 seed: {pending}")
        allres = [train_one(s, epochs=a.epochs) for s in pending]
        combined = dict(
            seeds=[r['seed'] for r in allres],
            test_miss=[r['test']['miss_rate'] for r in allres],
            test_review=[r['test']['review_rate'] for r in allres],
            test_pass=[100 - r['test']['review_rate'] for r in allres],
            head_acc=[r['head_acc'] for r in allres],
        )
        json.dump(combined, open(PROJECT_ROOT / 'data' / 'vit_tiny_all_seeds.json', 'w'),
                  ensure_ascii=False, indent=2)
        print("ALL SEEDS DONE:", json.dumps(combined, ensure_ascii=False))
    else:
        train_one(a.seed, epochs=a.epochs)


if __name__ == '__main__':
    main()
