#!/usr/bin/env python3
"""
run_patchcore_vpim.py  v4
--------------------------
修复：加入 PatchCore neighbourhood aggregation (3×3 avg pool)
这是 PatchCore 高性能的关键步骤，之前缺失导致 I-AUROC 虚低约 20%。

Usage:
  # 单类验证
  python run_patchcore_vpim.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --category agriculture/maize \
      --batch_size 4 --coreset 0.1

  # baseline 对照
  python run_patchcore_vpim.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --category agriculture/maize \
      --batch_size 4 --coreset 0.1 --baseline

  # 全量 38 类（后台）
  nohup python run_patchcore_vpim.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --all --batch_size 8 --coreset 0.1 \
      --output_dir /home/waas/results/vpim \
      > /home/waas/vpim_run.log 2>&1 &
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

sys.path.insert(0, str(Path(__file__).parent))
from vpim import VPIM


# ─────────────────────────────────────────────────────────────────────────────
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
    p.add_argument("--output_dir", type=str, default="./results/vpim")
    p.add_argument("--num_views",  type=int, default=5)
    p.add_argument("--img_size",   type=int, default=256,
                   help="Per-view image size. MANTA view is 256×256.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--coreset",    type=float, default=0.1)
    p.add_argument("--knn_k",      type=int, default=9,
                   help="k-NN neighbours for anomaly score (PatchCore default: 9)")
    p.add_argument("--nbhd",       type=int, default=3,
                   help="Neighbourhood aggregation kernel size (PatchCore default: 3)")
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--baseline",   action="store_true",
                   help="Bypass VPIM for diagnostic comparison")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MANTADataset(Dataset):
    """
    Splits wide image (H × N*W) into N per-view crops, resizes each to
    img_size × img_size, returns stacked tensor.

    Returns:
        views : (N, 3, H, H)
        label : int
        masks : (N, H, H)   float 0/1
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, category, split, img_size=256, num_views=5):
        self.cat_root  = Path(root) / category
        self.img_size  = img_size
        self.num_views = num_views
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size),
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
                mp = (self.cat_root / "ground_truth" /
                      cls_dir.name / img_path.name)
                self.samples.append((
                    img_path, int(is_ano),
                    mp if (is_ano and mp.exists()) else None
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        wide = Image.open(img_path).convert("RGB")
        W_view = wide.width // self.num_views

        views, masks = [], []
        for i in range(self.num_views):
            crop = wide.crop((i*W_view, 0, (i+1)*W_view, wide.height))
            views.append(self.img_tf(crop))
            if mask_path:
                mw = Image.open(mask_path).convert("L")
                mc = mw.crop((i*W_view, 0, (i+1)*W_view, mw.height))
                masks.append(
                    (self.mask_tf(mc) > 0.5).float().squeeze(0)
                )
            else:
                masks.append(torch.zeros(self.img_size, self.img_size))

        return torch.stack(views), label, torch.stack(masks)


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.tensor([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood aggregation (PatchCore core op)
# ─────────────────────────────────────────────────────────────────────────────

def neighbourhood_aggregate(f: torch.Tensor, p: int = 3) -> torch.Tensor:
    """
    PatchCore neighbourhood aggregation.
    Each patch token is replaced by the average of its p×p spatial neighbours.
    This encodes local context and is critical for PatchCore performance.

    Args:
        f: (B, C, H, W)
        p: kernel size (default 3, same as PatchCore paper)
    Returns:
        (B, C, H, W)
    """
    if p <= 1:
        return f
    return F.avg_pool2d(f, kernel_size=p, stride=1, padding=p // 2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor: per-view backbone + neighbourhood agg + VPIM
# ─────────────────────────────────────────────────────────────────────────────

class VPIMExtractor(nn.Module):
    """
    Pipeline:
      1. Each view → WideResNet-50 (frozen) → f2 (512), f3 (1024)
      2. Neighbourhood aggregation on each scale (PatchCore step)
      3. Concatenate views along width → wide feature map
      4. VPIM cross-view enhancement (or identity if baseline=True)
      5. Multi-scale concat (512+1024=1536)
      6. Flatten → patch tokens
    """

    def __init__(self, num_views=5, num_heads=4, nbhd=3, baseline=False):
        super().__init__()
        bb = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2   # 512 ch
        self.layer3 = bb.layer3   # 1024 ch
        for p in self.parameters():
            p.requires_grad_(False)

        self.vpim2     = VPIM(512,  num_views, num_heads)
        self.vpim3     = VPIM(1024, num_views, num_heads)
        self.num_views = num_views
        self.nbhd      = nbhd
        self.baseline  = baseline

    @torch.no_grad()
    def _backbone_one_view(self, x):
        """x: (B,3,H,H) → f2:(B,512,h,h), f3:(B,1024,h',h')"""
        x  = self.layer0(x)
        x  = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3

    def _enhance_wide(self, f_wide, vpim):
        """
        Apply split → cross-view attention → enhance (no aggregation).
        f_wide: (B,C,h,N*w) → (B,C,h,N*w)
        """
        B, C, h, Nw = f_wide.shape
        views = vpim.view_split(f_wide)               # B,N,C,h,w
        gap   = views.mean(dim=[-2, -1])              # B,N,C
        eq    = vpim.cross_view_attn(gap)             # B,N,C
        ev    = vpim.feature_enhance(views, eq)       # B,N,C,h,w
        return ev.permute(0,2,3,1,4).reshape(B,C,h,Nw)

    def forward(self, views: torch.Tensor):
        """
        Args:
            views: (B, N, 3, H, H)
        Returns:
            patches      : (B, N*h*w, 1536)
            features_wide: (B, 1536, h, N*w)
        """
        B, N = views.shape[:2]
        f2_list, f3_list = [], []
        for i in range(N):
            f2, f3 = self._backbone_one_view(views[:, i])
            # ★ neighbourhood aggregation per view per scale
            f2 = neighbourhood_aggregate(f2, self.nbhd)
            f3 = neighbourhood_aggregate(f3, self.nbhd)
            f2_list.append(f2)
            f3_list.append(f3)

        # Concatenate views along width
        f2_wide = torch.cat(f2_list, dim=-1)   # B,512,h,N*w
        f3_wide = torch.cat(f3_list, dim=-1)   # B,1024,h',N*w'

        if self.baseline:
            ev2 = f2_wide
            ev3 = f3_wide
        else:
            ev2 = self._enhance_wide(f2_wide, self.vpim2)
            ev3 = self._enhance_wide(f3_wide, self.vpim3)

        # Align scales
        ev3 = F.interpolate(ev3, size=ev2.shape[-2:],
                            mode="bilinear", align_corners=False)

        features_wide = torch.cat([ev2, ev3], dim=1)    # B,1536,h,N*w

        B2, C, fH, fW = features_wide.shape
        patches = features_wide.permute(0,2,3,1).reshape(B2, fH*fW, C)
        return patches, features_wide


# ─────────────────────────────────────────────────────────────────────────────
# PatchCore: memory bank + KNN scoring
# ─────────────────────────────────────────────────────────────────────────────

def build_memory_bank(model, loader, device, coreset_ratio, seed=42):
    model.eval()
    all_feats = []
    for views, *_ in tqdm(loader, desc="  Building memory bank"):
        with torch.no_grad():
            patches, _ = model(views.to(device))
        all_feats.append(patches.cpu().numpy().reshape(-1, patches.shape[-1]))
    all_feats = np.concatenate(all_feats, axis=0)
    k   = max(1, int(len(all_feats) * coreset_ratio))
    idx = np.random.RandomState(seed).choice(len(all_feats), k, replace=False)
    print(f"  Memory bank: {len(all_feats):,} → {k:,} patches "
          f"(coreset {coreset_ratio:.0%})")
    return all_feats[idx]


def run_inference(model, loader, memory, device, knn_k, img_size, num_views):
    nn_idx = NearestNeighbors(n_neighbors=knn_k, metric="euclidean", n_jobs=-1)
    nn_idx.fit(memory)

    model.eval()
    img_scores, img_labels, anomaly_maps, gt_masks = [], [], [], []

    for views, labels, masks in tqdm(loader, desc="  Inference"):
        with torch.no_grad():
            patches, feat_wide = model(views.to(device))

        B, HW, C = patches.shape
        _, _, fH, fW = feat_wide.shape

        dists, _ = nn_idx.kneighbors(patches.cpu().numpy().reshape(-1, C))
        # PatchCore image score: max over patches, mean over k neighbours
        patch_scores = dists.mean(axis=1).reshape(B, fH, fW)

        for b in range(B):
            img_scores.append(float(patch_scores[b].max()))
            img_labels.append(int(labels[b]))

            # Pixel-level map: upsample to H × N*H
            sm = torch.tensor(patch_scores[b]).unsqueeze(0).unsqueeze(0)
            sm = F.interpolate(sm,
                               size=(img_size, img_size * num_views),
                               mode="bilinear", align_corners=False)
            sm = gaussian_filter(sm.squeeze().numpy(), sigma=4)
            anomaly_maps.append(sm)

            # GT mask: (N,H,H) → concatenate to (H, N*H)
            m = masks[b].numpy()
            gt_masks.append(
                np.concatenate([m[i] for i in range(num_views)], axis=1)
            )

    return (np.array(img_scores), np.array(img_labels),
            np.array(anomaly_maps), np.array(gt_masks))


def compute_metrics(img_scores, img_labels, anomaly_maps, gt_masks):
    i_auroc = (roc_auc_score(img_labels, img_scores) * 100
               if len(np.unique(img_labels)) > 1 else float("nan"))
    flat_pred = anomaly_maps.flatten()
    flat_gt   = gt_masks.flatten().astype(int)
    p_auroc = (roc_auc_score(flat_gt, flat_pred) * 100
               if flat_gt.sum() > 0 else float("nan"))
    return i_auroc, p_auroc


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_category(args, category):
    device = args.device if torch.cuda.is_available() else "cpu"
    mode   = "baseline" if args.baseline else "VPIM"
    print(f"\n{'─'*58}\nCategory : {category}  [{mode}]\n{'─'*58}")

    try:
        train_ds = MANTADataset(args.data_root, category, "train",
                                args.img_size, args.num_views)
        test_ds  = MANTADataset(args.data_root, category, "test",
                                args.img_size, args.num_views)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {"category": category, "error": str(e)}

    kw = dict(shuffle=False, num_workers=4,
              pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, **kw)
    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

    model  = VPIMExtractor(args.num_views, args.num_heads,
                           args.nbhd, args.baseline).to(device)
    memory = build_memory_bank(model, train_loader, device, args.coreset)

    img_scores, img_labels, anomaly_maps, gt_masks = run_inference(
        model, test_loader, memory, device,
        args.knn_k, args.img_size, args.num_views
    )
    i_auroc, p_auroc = compute_metrics(img_scores, img_labels,
                                       anomaly_maps, gt_masks)

    print(f"\n  I-AUROC : {i_auroc:.1f}%")
    print(f"  P-AUROC : {p_auroc:.1f}%")
    return {
        "category": category,
        "mode"    : mode,
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
        with open(out_dir / "vpim_results.json", "w") as f:
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
    print(f"\nResults → {out_dir / 'vpim_results.json'}")


if __name__ == "__main__":
    main()
