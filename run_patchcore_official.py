#!/usr/bin/env python3
"""
run_patchcore_official.py
--------------------------
按照 MANTA 论文官方设置复现 PatchCore baseline：
  - 输入：224×1120 宽图整张送入 backbone（multi-view setting）
  - coreset: 0.01（论文设置，percentage 参数）
  - backbone: WideResNet-50, layer2 + layer3
  - neighbourhood aggregation: 3×3 avg pool（PatchCore 标准）
  - knn_k: 9（PatchCore 原论文默认）
  - img_size: 224（论文设置）

目标：复现论文 Table 3 的 PatchCore 结果（全局 95.0% I-AUROC）

Usage:
  # 单类验证
  python run_patchcore_official.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --category agriculture/maize

  # 全量 38 类
  nohup python run_patchcore_official.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --all \
      --output_dir /home/waas/results/patchcore_official \
      > /home/waas/patchcore_official.log 2>&1 &
"""

import sys, json, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter


MANTA_CATEGORIES = [
    "agriculture/maize",  "agriculture/paddy",   "agriculture/soybean", "agriculture/wheat",
    "electronics/block_inductor","electronics/copper_standoff","electronics/flat_nut",
    "electronics/led","electronics/led_pad","electronics/long_button",
    "electronics/power_inductor","electronics/short_button","electronics/thin_resistor",
    "electronics/type_c","electronics/wafer_resistor",
    "groceries/coffee_beans","groceries/goji_berries","groceries/pistachios",
    "mechanics/button","mechanics/gear","mechanics/nut","mechanics/nut_cap",
    "mechanics/red_washer","mechanics/round_button_cap","mechanics/screw",
    "mechanics/square_button_cap","mechanics/terminal","mechanics/wire_cap",
    "mechanics/yellow_green_washer",
    "medicine/capsule","medicine/coated_tablet","medicine/embossed_tablet",
    "medicine/lettered_tablet","medicine/oblong_tablet","medicine/pink_tablet",
    "medicine/red_tablet","medicine/white_tablet","medicine/yellow_tablet",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--category",   type=str, default="agriculture/maize")
    p.add_argument("--all",        action="store_true")
    p.add_argument("--output_dir", type=str, default="./results/patchcore_official")
    p.add_argument("--num_views",  type=int, default=5)
    # 论文官方设置
    p.add_argument("--img_h",      type=int, default=224,
                   help="Per-view height (论文: 224)")
    p.add_argument("--img_w_view", type=int, default=224,
                   help="Per-view width (论文: 224, total=1120)")
    p.add_argument("--coreset",    type=float, default=0.01,
                   help="论文设置: 0.01 (percentage parameter)")
    p.add_argument("--knn_k",      type=int, default=9,
                   help="PatchCore 原论文默认: 9")
    p.add_argument("--nbhd",       type=int, default=3,
                   help="Neighbourhood aggregation kernel (PatchCore: 3)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: 224×1120 宽图整张，和论文一致
# ─────────────────────────────────────────────────────────────────────────────

class MANTADataset(Dataset):
    """
    论文 multi-view setting:
      输入 224×1120 宽图（5个视角横向拼接），整张图送入 backbone。
      这与论文 Supp Section 3 描述一致：
        "The inputs were resized as 224×224 for single-view
         and 224×1120 for multi-view."
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, category, split,
                 img_h=224, img_w_view=224, num_views=5):
        self.cat_root  = Path(root) / category
        self.img_h     = img_h
        self.img_w     = img_w_view * num_views   # 224×5 = 1120
        self.num_views = num_views

        # 整张宽图 resize 到 224×1120
        self.img_tf = transforms.Compose([
            transforms.Resize((img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_h, self.img_w),
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.samples: List[Tuple] = []
        split_dir = self.cat_root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            is_ano = cls_dir.name != "good"
            for img_path in sorted(cls_dir.glob("*.png")):
                mp = self.cat_root/"ground_truth"/cls_dir.name/img_path.name
                self.samples.append((img_path, int(is_ano),
                                     mp if (is_ano and mp.exists()) else None))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img  = self.img_tf(Image.open(img_path).convert("RGB"))
        mask = (self.mask_tf(Image.open(mask_path).convert("L")) > 0.5).float() \
               if mask_path else torch.zeros(1, self.img_h, self.img_w)
        return img, label, mask


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore feature extractor: 宽图整张送入 backbone
# ─────────────────────────────────────────────────────────────────────────────

def neighbourhood_aggregate(f, p=3):
    if p <= 1: return f
    return F.avg_pool2d(f, kernel_size=p, stride=1, padding=p//2)


class PatchCoreExtractor(nn.Module):
    """
    标准 PatchCore 特征提取器。
    整张 224×1120 宽图送入 WideResNet-50，提取 layer2+layer3 特征。
    与论文官方实现一致。
    """
    def __init__(self, nbhd=3):
        super().__init__()
        bb = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2   # 512 ch
        self.layer3 = bb.layer3   # 1024 ch
        for p in self.parameters():
            p.requires_grad_(False)
        self.nbhd = nbhd

    def forward(self, x):
        """x: (B, 3, 224, 1120)"""
        x  = self.layer0(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)                    # B, 512,  28, 140
        f3 = self.layer3(f2)                   # B, 1024, 14, 70

        # neighbourhood aggregation（PatchCore 核心步骤）
        f2 = neighbourhood_aggregate(f2, self.nbhd)
        f3 = neighbourhood_aggregate(f3, self.nbhd)

        # 上采样 f3 到 f2 尺寸
        f3 = F.interpolate(f3, size=f2.shape[-2:],
                           mode="bilinear", align_corners=False)

        # 拼接多尺度特征: B, 1536, 28, 140
        features = torch.cat([f2, f3], dim=1)

        # 展平为 patch tokens: B, 28×140, 1536
        B, C, H, W = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return patches, features


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_memory_bank(model, loader, device, coreset_ratio, seed=42):
    model.eval(); all_feats = []
    for imgs, *_ in tqdm(loader, desc="  Memory bank"):
        with torch.no_grad():
            patches, _ = model(imgs.to(device))
        all_feats.append(patches.cpu().numpy().reshape(-1, patches.shape[-1]))
    all_feats = np.concatenate(all_feats, axis=0)
    k   = max(1, int(len(all_feats) * coreset_ratio))
    idx = np.random.RandomState(seed).choice(len(all_feats), k, replace=False)
    print(f"  Memory bank: {len(all_feats):,} → {k:,} patches "
          f"(coreset {coreset_ratio:.1%})")
    return all_feats[idx]


def run_inference(model, loader, memory, device, knn_k, img_h, img_w):
    nn_idx = NearestNeighbors(n_neighbors=knn_k, metric="euclidean", n_jobs=-1)
    nn_idx.fit(memory)
    model.eval()
    img_scores, img_labels, anomaly_maps, gt_masks = [], [], [], []

    for imgs, labels, masks in tqdm(loader, desc="  Inference"):
        with torch.no_grad():
            patches, feat = model(imgs.to(device))
        B, HW, C = patches.shape
        _, _, fH, fW = feat.shape
        dists, _ = nn_idx.kneighbors(patches.cpu().numpy().reshape(-1, C))
        ps = dists.mean(axis=1).reshape(B, fH, fW)

        for b in range(B):
            img_scores.append(float(ps[b].max()))
            img_labels.append(int(labels[b]))
            sm = F.interpolate(
                torch.tensor(ps[b]).unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear", align_corners=False)
            anomaly_maps.append(gaussian_filter(sm.squeeze().numpy(), sigma=4))
            gt_masks.append(masks[b, 0].numpy())

    return (np.array(img_scores), np.array(img_labels),
            np.array(anomaly_maps), np.array(gt_masks))


def compute_metrics(img_scores, img_labels, anomaly_maps, gt_masks):
    i = (roc_auc_score(img_labels, img_scores)*100
         if len(np.unique(img_labels)) > 1 else float("nan"))
    flat_pred = anomaly_maps.flatten()
    flat_gt   = gt_masks.flatten().astype(int)
    p = (roc_auc_score(flat_gt, flat_pred)*100
         if flat_gt.sum() > 0 else float("nan"))
    return i, p


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_category(args, category):
    device = args.device if torch.cuda.is_available() else "cpu"
    img_w  = args.img_w_view * args.num_views   # 1120
    print(f"\n{'─'*58}\nCategory : {category}\n{'─'*58}")

    try:
        train_ds = MANTADataset(args.data_root, category, "train",
                                args.img_h, args.img_w_view, args.num_views)
        test_ds  = MANTADataset(args.data_root, category, "test",
                                args.img_h, args.img_w_view, args.num_views)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {"category": category, "error": str(e)}

    kw = dict(shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, **kw)
    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"  Input size: {args.img_h}×{img_w} (论文官方设置)")
    print(f"  Coreset: {args.coreset} | knn_k: {args.knn_k}")

    model  = PatchCoreExtractor(args.nbhd).to(device)
    memory = build_memory_bank(model, train_loader, device, args.coreset)
    img_scores, img_labels, anomaly_maps, gt_masks = run_inference(
        model, test_loader, memory, device,
        args.knn_k, args.img_h, img_w)
    i_auroc, p_auroc = compute_metrics(img_scores, img_labels,
                                       anomaly_maps, gt_masks)
    print(f"\n  I-AUROC : {i_auroc:.1f}%")
    print(f"  P-AUROC : {p_auroc:.1f}%")
    return {
        "category": category,
        "i_auroc" : round(i_auroc, 1) if not np.isnan(i_auroc) else None,
        "p_auroc" : round(p_auroc, 1) if not np.isnan(p_auroc) else None,
    }


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = MANTA_CATEGORIES if args.all else [args.category]
    results = []

    for cat in categories:
        res = run_category(args, cat)
        results.append(res)
        with open(out_dir / "patchcore_official_results.json", "w") as f:
            json.dump(results, f, indent=2)

    valid = [r for r in results if r.get("i_auroc") is not None]
    if valid:
        print(f"\n{'═'*58}")
        print(f"{'Category':<38} {'I-AUROC':>8} {'P-AUROC':>8}")
        print(f"{'─'*58}")
        for r in valid:
            print(f"{r['category']:<38} "
                  f"{str(r['i_auroc'])+'%':>8} "
                  f"{str(r['p_auroc'])+'%':>8}")
        i_mean = np.mean([r["i_auroc"] for r in valid])
        p_vals  = [r["p_auroc"] for r in valid if r.get("p_auroc")]
        p_mean  = np.mean(p_vals) if p_vals else float("nan")
        print(f"{'─'*58}")
        print(f"{'Mean ('+str(len(valid))+' cat)':<38} "
              f"{i_mean:>7.1f}% {p_mean:>7.1f}%")
        print(f"\n论文 Table 3 PatchCore 全局均值: 95.0% / 95.7%")
    print(f"\nResults → {out_dir / 'patchcore_official_results.json'}")


if __name__ == "__main__":
    main()
