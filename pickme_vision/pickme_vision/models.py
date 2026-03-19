from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 8) -> None:
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(groups, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvGNAct(3, 16),
            ConvGNAct(16, 16),
            nn.MaxPool2d(2),
            ConvGNAct(16, 32),
            ConvGNAct(32, 32),
            nn.MaxPool2d(2),
            ConvGNAct(32, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResidualBlockGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, kernel_size=3, stride=stride)
        num_groups = min(8, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.act(out)


class SmallResNetGN(nn.Module):
    def __init__(self, num_classes: int = 10, widths: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        self.stem = ConvGNAct(3, widths[0])
        self.layer1 = nn.Sequential(
            ResidualBlockGN(widths[0], widths[0], stride=1),
            ResidualBlockGN(widths[0], widths[0], stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlockGN(widths[0], widths[1], stride=2),
            ResidualBlockGN(widths[1], widths[1], stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlockGN(widths[1], widths[2], stride=2),
            ResidualBlockGN(widths[2], widths[2], stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        dw_groups = in_ch
        pw_groups = min(8, out_ch)
        while out_ch % pw_groups != 0 and pw_groups > 1:
            pw_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=dw_groups, bias=False),
            nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=pw_groups, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = ConvGNAct(3, 16)
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(16, 32, stride=1),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64, stride=1),
            DepthwiseSeparableBlock(64, 80, stride=2),
            DepthwiseSeparableBlock(80, 96, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


MODEL_FACTORY: dict[str, Callable[[int], nn.Module]] = {
    "simplecnn": lambda num_classes: SimpleCNN(num_classes=num_classes),
    "resnet_gn": lambda num_classes: SmallResNetGN(num_classes=num_classes),
    "depthwisecnn": lambda num_classes: DepthwiseCNN(num_classes=num_classes),
    "tinyvit": lambda num_classes: TinyViT(num_classes=num_classes),
}



def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    key = model_name.lower()
    if key not in MODEL_FACTORY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_FACTORY)}")
    return MODEL_FACTORY[key](num_classes)
