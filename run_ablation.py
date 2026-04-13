#!/usr/bin/env python3
"""
run_ablation.py — VPIM Ablation Study (v2)
------------------------------------------
5种消融配置：
  A_baseline   纯PatchCore backbone，无VPIM
  B_split_only 视角分割+pos_embed，无跨视角注意力
  C_no_enhance split+跨视角注意力，无FeatureEnhancement
  D_no_pos     完整VPIM，pos_embed归零
  E_full_vpim  完整VPIM（正式结果）

全量38类运行命令：
  nohup python run_ablation.py \
      --data_root /home/waas/data/MANTA-Tiny \
      --all --batch_size 8 --coreset 0.1 \
      --knn_k 9 --nbhd 3 \
      --output_dir /home/waas/results/ablation \
      > /home/waas/ablation.log 2>&1 &
"""

import sys, json, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from vpim import VPIM


MANTA_CATEGORIES = [
    "agriculture/maize","agriculture/paddy","agriculture/soybean","agriculture/wheat",
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

ABLATION_CONFIGS = {
    "A_baseline":   "纯PatchCore backbone，无VPIM",
    "B_split_only": "视角分割+pos_embed，无跨视角注意力",
    "C_no_enhance": "split+跨视角注意力，无FeatureEnhancement",
    "D_no_pos":     "完整VPIM，pos_embed归零",
    "E_full_vpim":  "完整VPIM（正式结果）",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--category",   type=str, default="agriculture/maize")
    p.add_argument("--all",        action="store_true",
                   help="Run all 38 MANTA-Tiny categories")
    p.add_argument("--output_dir", type=str, default="./results/ablation")
    p.add_argument("--num_views",  type=int, default=5)
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--coreset",    type=float, default=0.1)
    p.add_argument("--knn_k",      type=int, default=9)
    p.add_argument("--nbhd",       type=int, default=3)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--configs",    type=str, default="all",
                   help="Comma-separated configs, e.g. A_baseline,E_full_vpim")
    return p.parse_args()


class MANTADataset(torch.utils.data.Dataset):
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
                mp = self.cat_root/"ground_truth"/cls_dir.name/img_path.name
                self.samples.append((img_path, int(is_ano),
                                     mp if (is_ano and mp.exists()) else None))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        wide   = Image.open(img_path).convert("RGB")
        W_view = wide.width // self.num_views
        views, masks = [], []
        for i in range(self.num_views):
            crop = wide.crop((i*W_view, 0, (i+1)*W_view, wide.height))
            views.append(self.img_tf(crop))
            if mask_path:
                mw = Image.open(mask_path).convert("L")
                mc = mw.crop((i*W_view, 0, (i+1)*W_view, mw.height))
                masks.append((self.mask_tf(mc) > 0.5).float().squeeze(0))
            else:
                masks.append(torch.zeros(self.img_size, self.img_size))
        return torch.stack(views), label, torch.stack(masks)


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def neighbourhood_aggregate(f, p=3):
    if p <= 1:
        return f
    return F.avg_pool2d(f, kernel_size=p, stride=1, padding=p // 2)


class AblationExtractor(nn.Module):
    def __init__(self, config, num_views=5, num_heads=4, nbhd=3):
        super().__init__()
        assert config in ABLATION_CONFIGS, f"Unknown config: {config}"
        self.config    = config
        self.num_views = num_views
        self.nbhd      = nbhd

        bb = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        for p in self.parameters():
            p.requires_grad_(False)

        self.vpim2 = VPIM(512,  num_views, num_heads)
        self.vpim3 = VPIM(1024, num_views, num_heads)

        if config == "D_no_pos":
            nn.init.zeros_(self.vpim2.view_split.pos_embed)
            nn.init.zeros_(self.vpim3.view_split.pos_embed)

    @torch.no_grad()
    def _backbone(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3

    def _forward_one_scale(self, f_wide, vpim):
        B, C, h, Nw = f_wide.shape
        cfg = self.config

        if cfg == "A_baseline":
            return f_wide

        views = vpim.view_split(f_wide)               # B,N,C,h,w

        if cfg == "B_split_only":
            out = views.mean(dim=1)                   # B,C,h,w
            w   = Nw // self.num_views
            out = out.unsqueeze(3).expand(B, C, h, self.num_views, w)
            return out.reshape(B, C, h, Nw)

        gap = views.mean(dim=[-2, -1])                # B,N,C
        eq  = vpim.cross_view_attn(gap)               # B,N,C

        if cfg == "C_no_enhance":
            w_scores = vpim.aggregation.score_head(eq).squeeze(-1)
            weights  = F.softmax(w_scores, dim=-1).view(B, self.num_views, 1, 1, 1)
            out = (views * weights).sum(dim=1)
            w   = Nw // self.num_views
            out = out.unsqueeze(3).expand(B, C, h, self.num_views, w)
            return out.reshape(B, C, h, Nw)

        # D_no_pos / E_full_vpim: full pipeline
        ev = vpim.feature_enhance(views, eq)          # B,N,C,h,w
        w  = Nw // self.num_views
        return ev.permute(0, 2, 3, 1, 4).reshape(B, C, h, Nw)

    def forward(self, views_in):
        B, N = views_in.shape[:2]
        f2_list, f3_list = [], []
        for i in range(N):
            f2, f3 = self._backbone(views_in[:, i])
            f2_list.append(neighbourhood_aggregate(f2, self.nbhd))
            f3_list.append(neighbourhood_aggregate(f3, self.nbhd))

        f2_wide = torch.cat(f2_list, dim=-1)
        f3_wide = torch.cat(f3_list, dim=-1)

        ev2 = self._forward_one_scale(f2_wide, self.vpim2)
        ev3 = self._forward_one_scale(f3_wide, self.vpim3)
        ev3 = F.interpolate(ev3, size=ev2.shape[-2:],
                            mode="bilinear", align_corners=False)
        features = torch.cat([ev2, ev3], dim=1)       # B,1536,h,Nw
        B2, C, fH, fW = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(B2, fH * fW, C)
        return patches, features


def build_memory_bank(model, loader, device, coreset_ratio, seed=42):
    model.eval()
    all_feats = []
    for views, *_ in tqdm(loader, desc="  Memory bank", leave=False):
        with torch.no_grad():
            patches, _ = model(views.to(device))
        all_feats.append(patches.cpu().numpy().reshape(-1, patches.shape[-1]))
    all_feats = np.concatenate(all_feats, axis=0)
    k   = max(1, int(len(all_feats) * coreset_ratio))
    idx = np.random.RandomState(seed).choice(len(all_feats), k, replace=False)
    return all_feats[idx]


def run_inference(model, loader, memory, device, knn_k, img_size, num_views):
    nn_idx = NearestNeighbors(n_neighbors=knn_k, metric="euclidean", n_jobs=-1)
    nn_idx.fit(memory)
    model.eval()
    img_scores, img_labels, anomaly_maps, gt_masks = [], [], [], []
    for views, labels, masks in tqdm(loader, desc="  Inference", leave=False):
        with torch.no_grad():
            patches, feat = model(views.to(device))
        B, HW, C = patches.shape
        _, _, fH, fW = feat.shape
        dists, _ = nn_idx.kneighbors(patches.cpu().numpy().reshape(-1, C))
        ps = dists.mean(axis=1).reshape(B, fH, fW)
        for b in range(B):
            img_scores.append(float(ps[b].max()))
            img_labels.append(int(labels[b]))
            sm = F.interpolate(
                torch.tensor(ps[b]).unsqueeze(0).unsqueeze(0),
                size=(img_size, img_size * num_views),
                mode="bilinear", align_corners=False)
            anomaly_maps.append(gaussian_filter(sm.squeeze().numpy(), sigma=4))
            m = masks[b].numpy()
            gt_masks.append(
                np.concatenate([m[i] for i in range(num_views)], axis=1))
    return (np.array(img_scores), np.array(img_labels),
            np.array(anomaly_maps), np.array(gt_masks))


def compute_metrics(img_scores, img_labels, anomaly_maps, gt_masks):
    i = (roc_auc_score(img_labels, img_scores) * 100
         if len(np.unique(img_labels)) > 1 else float("nan"))
    flat_pred = anomaly_maps.flatten()
    flat_gt   = gt_masks.flatten().astype(int)
    p = (roc_auc_score(flat_gt, flat_pred) * 100
         if flat_gt.sum() > 0 else float("nan"))
    return i, p


def run_one(args, category, config):
    device = args.device if torch.cuda.is_available() else "cpu"
    try:
        train_ds = MANTADataset(args.data_root, category, "train",
                                args.img_size, args.num_views)
        test_ds  = MANTADataset(args.data_root, category, "test",
                                args.img_size, args.num_views)
    except FileNotFoundError as e:
        return {"category": category, "config": config, "error": str(e)}

    kw = dict(shuffle=False, num_workers=4,
              pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, **kw)

    model  = AblationExtractor(config, args.num_views,
                               args.num_heads, args.nbhd).to(device)
    memory = build_memory_bank(model, train_loader, device, args.coreset)
    scores, labels, amaps, gmasks = run_inference(
        model, test_loader, memory, device,
        args.knn_k, args.img_size, args.num_views)
    i_auroc, p_auroc = compute_metrics(scores, labels, amaps, gmasks)
    return {
        "category": category,
        "config"  : config,
        "desc"    : ABLATION_CONFIGS[config],
        "i_auroc" : round(i_auroc, 1) if not np.isnan(i_auroc) else None,
        "p_auroc" : round(p_auroc, 1) if not np.isnan(p_auroc) else None,
    }


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = MANTA_CATEGORIES if args.all else [args.category]
    configs    = (list(ABLATION_CONFIGS.keys()) if args.configs == "all"
                  else [c.strip() for c in args.configs.split(",")])
    results = []

    for cat in categories:
        print(f"\n{'═'*60}\nCategory: {cat}\n{'═'*60}")
        for cfg in configs:
            print(f"\n  [{cfg}] {ABLATION_CONFIGS[cfg]}")
            res = run_one(args, cat, cfg)
            results.append(res)
            if res.get("i_auroc"):
                print(f"    I-AUROC: {res['i_auroc']}%  "
                      f"P-AUROC: {res['p_auroc']}%")
            with open(out_dir / "ablation_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Summary: mean per config across all categories
    valid = [r for r in results if r.get("i_auroc") is not None]
    if valid:
        print(f"\n{'═'*65}")
        print(f"{'Config':<18} {'I-AUROC':>9} {'P-AUROC':>9}  描述")
        print(f"{'─'*65}")
        for cfg in configs:
            cfg_res = [r for r in valid if r["config"] == cfg]
            if not cfg_res:
                continue
            i_mean = np.mean([r["i_auroc"] for r in cfg_res])
            p_vals = [r["p_auroc"] for r in cfg_res if r.get("p_auroc")]
            p_mean = np.mean(p_vals) if p_vals else float("nan")
            print(f"{cfg:<18} {i_mean:>8.1f}% {p_mean:>8.1f}%  "
                  f"{ABLATION_CONFIGS[cfg]}")
        print(f"{'═'*65}")

    print(f"\nResults → {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()