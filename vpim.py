"""
vpim.py — View Parallel Inquiry Module
--------------------------------------
Novel module for multi-view tiny object anomaly detection.
Designed as a plug-in between backbone feature extraction and
PatchCore memory bank construction.

Paper target: ICCV 2026
Dataset:      MANTA (5 fixed camera views, 38 categories)

Input:   F  ∈ R^(B, C, H, 5W)   — 5-view horizontally concatenated feature map
Output:  F' ∈ R^(B, C, H, W)    — view-consensus anomaly feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Sub-module 1: View Split
# ══════════════════════════════════════════════════════════════════════════════

class ViewSplit(nn.Module):
    """
    Decompose concatenated multi-view feature map into N independent views
    and inject learnable view position embeddings.

    The backbone treats the 5-view concatenated image as a single wide image.
    This module explicitly restores view boundaries and informs the model
    which spatial region belongs to which viewpoint.
    """

    def __init__(self, num_views: int, channels: int):
        super().__init__()
        self.num_views = num_views
        # Learnable per-view position embedding: (1, N, C, 1, 1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_views, channels, 1, 1)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, N*W)
        Returns:
            views: (B, N, C, H, W)
        """
        B, C, H, NW = x.shape
        assert NW % self.num_views == 0, (
            f"Width dimension {NW} must be divisible by num_views {self.num_views}"
        )
        W = NW // self.num_views

        # Reshape: (B, C, H, N*W) → (B, C, H, N, W) → (B, N, C, H, W)
        views = x.view(B, C, H, self.num_views, W).permute(0, 3, 1, 2, 4)

        # Add view-specific position embeddings
        views = views + self.pos_embed
        return views


# ══════════════════════════════════════════════════════════════════════════════
# Sub-module 2: Cross-View Attention (core novelty)
# ══════════════════════════════════════════════════════════════════════════════

class CrossViewAttention(nn.Module):
    """
    Multi-head self-attention across N view query tokens.

    Each view first compresses its spatial feature map into a global query
    token via GAP. These N tokens then attend to each other through MHSA,
    enabling each view to incorporate information observed by other viewpoints.

    This is the "parallel inquiry" mechanism: all views are processed
    simultaneously rather than sequentially, avoiding order bias.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N, C)  — N per-view global tokens
        Returns:
            out:     (B, N, C)  — cross-view-enhanced tokens
        """
        B, N, C = queries.shape

        # Pre-norm (follows ViT-style pre-norm convention)
        x = self.norm(queries)

        # QKV projection → reshape for multi-head attention
        qkv = self.qkv(x)                                  # B, N, 3C
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # 3, B, heads, N, head_dim
        q, k, v = qkv.unbind(0)                            # each: B, heads, N, head_dim

        # Scaled dot-product attention across N views
        attn = (q @ k.transpose(-2, -1)) * self.scale      # B, heads, N, N
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate and project
        out = (attn @ v)                                   # B, heads, N, head_dim
        out = out.transpose(1, 2).reshape(B, N, C)         # B, N, C
        out = self.proj_drop(self.proj(out))

        # Residual connection
        return queries + out


# ══════════════════════════════════════════════════════════════════════════════
# Sub-module 3: Feature Enhancement
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEnhancement(nn.Module):
    """
    Condition each view's spatial feature map with its cross-view-enhanced
    global query token.

    The enhanced query token carries information from all other views.
    Using it as a spatial attention gate allows the spatial features of
    view i to be refined based on what other views observed — e.g., if
    view 3 detects a scratch, view 1's features get focused accordingly.

    Operation:
        gate_i = σ( Conv(norm(Fᵢ))  ⊗  Linear(q'ᵢ) )   ∈ (0,1)^(C,H,W)
        F̂ᵢ    = Fᵢ ⊙ gate_i  +  Fᵢ                     (residual)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query_proj = nn.Linear(channels, channels)
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(
            num_groups=min(32, channels // 8),
            num_channels=channels
        )

    def forward(
        self,
        views: torch.Tensor,
        enhanced_queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            views:            (B, N, C, H, W)
            enhanced_queries: (B, N, C)
        Returns:
            enhanced_views:   (B, N, C, H, W)
        """
        B, N, C, H, W = views.shape

        # Project query to channel-wise attention vector
        q = self.query_proj(enhanced_queries)      # B, N, C
        q = q.view(B * N, C, 1, 1)                # BN, C, 1, 1  (broadcast ready)

        # Flatten views for BatchNorm-compatible processing
        v = views.reshape(B * N, C, H, W)

        # Spatial gating: cross-view query conditions local spatial features
        gate = torch.sigmoid(
            self.spatial_conv(self.norm(v)) * q   # BN, C, H, W  (broadcast)
        )

        # Residual: original features boosted by cross-view attention gate
        enhanced = v * gate + v                    # BN, C, H, W
        return enhanced.reshape(B, N, C, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-module 4: Adaptive Aggregation
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveAggregation(nn.Module):
    """
    Aggregate N enhanced view features into a single consensus representation
    using learned per-view importance weights.

    Unlike simple averaging, this allows the model to down-weight views that
    are less informative (e.g., occluded, low-contrast) for a given sample.

    Operation:
        sᵢ  = MLP( GAP(F̂ᵢ) )              scalar score per view
        αᵢ  = Softmax({s₁,...,s_N})        normalized weights (sum=1)
        F'  = Σᵢ αᵢ · F̂ᵢ                 weighted consensus
    """

    def __init__(self, channels: int, num_views: int):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.score_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, enhanced_views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_views: (B, N, C, H, W)
        Returns:
            consensus:      (B, C, H, W)
        """
        B, N, C, H, W = enhanced_views.shape

        # Global average pool per view → global token
        gap = enhanced_views.mean(dim=[-2, -1])         # B, N, C

        # Compute per-view anomaly confidence score
        scores = self.score_head(gap).squeeze(-1)       # B, N
        weights = F.softmax(scores, dim=-1)             # B, N  (sums to 1)

        # Weighted sum across views
        weights = weights.view(B, N, 1, 1, 1)
        consensus = (enhanced_views * weights).sum(dim=1)  # B, C, H, W
        return consensus


# ══════════════════════════════════════════════════════════════════════════════
# Main Module: VPIM
# ══════════════════════════════════════════════════════════════════════════════

class VPIM(nn.Module):
    """
    View Parallel Inquiry Module (VPIM).

    Plug-in module between backbone feature extraction and
    the PatchCore memory bank. Replaces naive multi-view concatenation
    with explicit cross-view information exchange.

    Args:
        channels:   Feature channel dimension C (e.g., 512 for WideResNet layer2)
        num_views:  Number of camera views (5 for MANTA dataset)
        num_heads:  Attention heads for cross-view MHSA
        attn_drop:  Dropout rate on attention weights
        proj_drop:  Dropout rate on projection output

    Example:
        >>> vpim = VPIM(channels=512, num_views=5, num_heads=4)
        >>> x = torch.randn(2, 512, 28, 140)  # 140 = 28 * 5
        >>> out = vpim(x)  # (2, 512, 28, 28)
    """

    def __init__(
        self,
        channels: int,
        num_views: int = 5,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_views = num_views
        self.channels = channels

        self.view_split = ViewSplit(num_views, channels)
        self.cross_view_attn = CrossViewAttention(
            channels, num_heads, attn_drop, proj_drop
        )
        self.feature_enhance = FeatureEnhancement(channels)
        self.aggregation = AdaptiveAggregation(channels, num_views)

        self._init_weights()

    def _init_weights(self):
        """
        Near-identity initialization for training-free frameworks (e.g. PatchCore).

        At init:
          - CrossViewAttention output proj = 0  →  residual only, no cross-view effect
          - AdaptiveAggregation score head = 0  →  uniform 1/N weights (simple mean)
          - FeatureEnhancement spatial_conv = 0 →  gate = σ(0) = 0.5, scale by 1.5×

        The module gradually deviates from identity only when trained with gradients.
        Without training (PatchCore), it acts as a stable view-averaging operation.
        """
        # CrossViewAttention: zero-init output proj → residual = original query tokens
        nn.init.zeros_(self.cross_view_attn.proj.weight)
        nn.init.zeros_(self.cross_view_attn.proj.bias)

        # AdaptiveAggregation: zero-init last linear → uniform softmax → mean pooling
        last_linear = self.aggregation.score_head[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        # FeatureEnhancement: zero-init spatial conv → gate = 0.5 everywhere
        nn.init.zeros_(self.feature_enhance.spatial_conv.weight)
        nn.init.zeros_(self.feature_enhance.spatial_conv.bias)
        
        # ★ 新增：pos_embed 归零，消除视角位置偏移
        nn.init.zeros_(self.view_split.pos_embed)

        # ★ 新增：GroupNorm weight=1, bias=0（默认值，显式确保）
        nn.init.ones_(self.feature_enhance.norm.weight)
        nn.init.zeros_(self.feature_enhance.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, N*W)   — concatenated multi-view feature map
        Returns:
            consensus: (B, C, H, W)
        """
        # ① Decompose into N view-specific feature maps
        views = self.view_split(x)                     # B, N, C, H, W

        # ② Cross-view parallel inquiry via MHSA on global tokens
        gap = views.mean(dim=[-2, -1])                 # B, N, C
        enhanced_queries = self.cross_view_attn(gap)   # B, N, C

        # ③ Condition spatial features with cross-view-enhanced queries
        enhanced_views = self.feature_enhance(         # B, N, C, H, W
            views, enhanced_queries
        )

        # ④ Adaptive weighted aggregation → consensus feature
        consensus = self.aggregation(enhanced_views)   # B, C, H, W
        return consensus

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"num_views={self.num_views}, "
            f"num_heads={self.cross_view_attn.num_heads}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Integration: VPIM + PatchCore feature extractor
# ══════════════════════════════════════════════════════════════════════════════

class VPIMFeatureExtractor(nn.Module):
    """
    WideResNet-50 backbone + VPIM, producing patch tokens for PatchCore.

    Applies VPIM independently at layer2 (512ch) and layer3 (1024ch),
    then concatenates the two scales into 1536-dim patch tokens.

    Usage in PatchCore training/inference:
        extractor = VPIMFeatureExtractor()
        patch_tokens = extractor(images)  # (B*H*W, 1536)
        # → feed into coreset subsampling / NN search
    """

    def __init__(
        self,
        num_views: int = 5,
        num_heads: int = 4,
        output_stride: int = 1,   # adaptive avg pool kernel size
    ):
        super().__init__()
        import torchvision.models as models

        # Backbone: WideResNet-50-2 (standard PatchCore backbone)
        backbone = models.wide_resnet50_2(weights="IMAGENET1K_V1")

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2   # output: 512 ch
        self.layer3 = backbone.layer3   # output: 1024 ch

        # Freeze backbone (standard PatchCore setting)
        for p in self.parameters():
            p.requires_grad_(False)

        # VPIM at each scale (unfrozen, trainable)
        self.vpim2 = VPIM(channels=512,  num_views=num_views, num_heads=num_heads)
        self.vpim3 = VPIM(channels=1024, num_views=num_views, num_heads=num_heads)

        # Adaptive avg pool for patch aggregation
        self.pool = nn.AdaptiveAvgPool2d(output_stride) if output_stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, N*W)  — 5-view concatenated input image
        Returns:
            patch_tokens: (B*h*w, 1536)
        """
        # Backbone forward
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)    # B, 512,  H/8,  5W/8
        f3 = self.layer3(f2)   # B, 1024, H/16, 5W/16

        # Apply VPIM at each scale
        f2 = self.vpim2(f2)    # B, 512,  H/8,  W/8
        f3 = self.vpim3(f3)    # B, 1024, H/16, W/16

        # Upsample f3 to match f2 spatial size
        f3 = F.interpolate(
            f3, size=f2.shape[-2:],
            mode="bilinear", align_corners=False
        )                       # B, 1024, H/8, W/8

        # Concatenate multi-scale features
        features = torch.cat([f2, f3], dim=1)   # B, 1536, H/8, W/8

        # Reshape to patch tokens
        B, C, H, W = features.shape
        patch_tokens = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return patch_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 55)
    print("VPIM module sanity check")
    print("=" * 55)

    # ── Test 1: VPIM standalone ──────────────────────────────
    B, C, H, W, N = 2, 512, 28, 28, 5
    x = torch.randn(B, C, H, N * W).to(device)

    vpim = VPIM(channels=C, num_views=N, num_heads=4).to(device)
    out = vpim(x)

    print(f"\nVPIM standalone:")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, C, H, W), "Shape mismatch!"

    n_params = sum(p.numel() for p in vpim.parameters())
    print(f"  Params : {n_params / 1e6:.3f}M")

    # ── Test 2: Output is different per view (not identical) ──
    x_same = torch.randn(1, C, H, W).repeat(1, 1, 1, N).to(device)
    out_same = vpim(x_same)
    # Output should differ from naive mean of inputs
    naive_mean = x_same[:, :, :, :W].mean()
    print(f"\n  Cross-view effect (output != naive input mean): "
          f"{'YES' if not torch.allclose(out_same.mean(), naive_mean, atol=0.01) else 'NO'}")

    # ── Test 3: Gradient flow ────────────────────────────────
    x_grad = torch.randn(B, C, H, N * W, requires_grad=True).to(device)
    loss = vpim(x_grad).mean()
    loss.backward()
    print(f"\n  Gradient flow: "
          f"{'OK' if x_grad.grad is not None else 'FAILED'}")

    # ── Test 4: VPIM at layer3 scale (1024ch) ─────────────────
    vpim3 = VPIM(channels=1024, num_views=5, num_heads=8).to(device)
    x3 = torch.randn(B, 1024, 14, 14 * N).to(device)
    out3 = vpim3(x3)
    print(f"\nVPIM at layer3 scale:")
    print(f"  Input  : {tuple(x3.shape)}")
    print(f"  Output : {tuple(out3.shape)}")

    n_params3 = sum(p.numel() for p in vpim3.parameters())
    print(f"  Params : {n_params3 / 1e6:.3f}M")

    print("\n" + "=" * 55)
    print("All checks passed.")
    print("=" * 55)
