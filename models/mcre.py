
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import BASE_CHANNELS, ENCODER_CHANNELS, FEATURE_DIM


def GN(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def checkpoint_forward(module, x):
    if module.training:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(module, x, use_reentrant=False)
    return module(x)


class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False
            ),
            GN(in_channels),
            nn.SiLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels=FEATURE_DIM):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.project(x)


class FactorizedConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=stride,
                padding=(0, 1, 1),
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )
        self.view = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.spatial(x)
        return self.view(x)


class ResidualFactorizedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 24)
        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),
            GN(hidden),
            nn.SiLU(inplace=True),
            FactorizedConv3D(hidden, hidden),
            nn.Conv3d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            ),
            GN(channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity
        return self.act(out)


class CrossViewGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.gate = nn.Sequential(
            nn.Conv3d(
                channels,
                channels // 4,
                kernel_size=1,
                bias=False
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                channels // 4,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.gate(w)
        return x * w + x


class EfficientMultiViewBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = ResidualFactorizedBlock(channels)
        self.cross = CrossViewGate(channels)

    def forward(self, x):
        x = self.residual(x)
        return self.cross(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = FactorizedConv3D(
            in_channels,
            out_channels,
            stride=(1, 2, 2)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 2, 2),
                bias=False
            ),
            GN(out_channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class EncoderStage(nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                EfficientMultiViewBlock(channels)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return checkpoint_forward(self.blocks, x)


class EncoderStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            FactorizedConv3D(
                3,
                BASE_CHANNELS,
                stride=(1, 2, 2)
            ),
            FactorizedConv3D(
                BASE_CHANNELS,
                BASE_CHANNELS
            )
        )

    def forward(self, x):
        return checkpoint_forward(self.block, x)


class Stage1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = EncoderStem()
        self.stage1 = EncoderStage(BASE_CHANNELS, depth=2)
        self.down1 = DownsampleBlock(
            BASE_CHANNELS,
            ENCODER_CHANNELS[1]
        )
        self.stage2 = EncoderStage(
            ENCODER_CHANNELS[1],
            depth=2
        )
        self.down2 = DownsampleBlock(
            ENCODER_CHANNELS[1],
            ENCODER_CHANNELS[2]
        )
        self.stage3 = EncoderStage(
            ENCODER_CHANNELS[2],
            depth=3
        )
        self.down3 = DownsampleBlock(
            ENCODER_CHANNELS[2],
            ENCODER_CHANNELS[3]
        )
        self.stage4 = EncoderStage(
            ENCODER_CHANNELS[3],
            depth=2
        )
        initialize_weights(self)

    def forward(self, x, debug=False):
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        P1 = self.stem(x)
        P1 = self.stage1(P1)

        P2 = self.down1(P1)
        P2 = self.stage2(P2)

        P3 = self.down2(P2)
        P3 = self.stage3(P3)

        P4 = self.down3(P3)
        P4 = self.stage4(P4)

        outputs = {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "P4": P4
        }

        if debug:
            print("=" * 70)
            print("Stage 1 Encoder")
            print("=" * 70)
            for k, v in outputs.items():
                print(f"{k:<3} : {tuple(v.shape)}")
            print("=" * 70)

        return outputs


class ScaleProjection(nn.Module):
    def __init__(self, in_channels, out_channels=FEATURE_DIM):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            GN(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return checkpoint_forward(self.project, x)


class AdaptiveScaleWeights(nn.Module):
    def __init__(self, channels=FEATURE_DIM):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                hidden,
                1,
                kernel_size=1,
                bias=True
            )
        )

    def forward(self, p2, p3, p4):
        s2 = self.fc(self.pool(p2))
        s3 = self.fc(self.pool(p3))
        s4 = self.fc(self.pool(p4))
        scores = torch.cat([s2, s3, s4], dim=1)
        return torch.softmax(scores, dim=1)


class SharedMultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.p2_proj = ScaleProjection(96)
        self.p3_proj = ScaleProjection(192)
        self.p4_proj = ScaleProjection(384)
        self.weight_predictor = AdaptiveScaleWeights()
        self.refine = nn.Sequential(
            FactorizedConv3D(FEATURE_DIM, FEATURE_DIM),
            FactorizedConv3D(FEATURE_DIM, FEATURE_DIM)
        )
        self.recognition_projection = nn.Sequential(
            nn.Conv3d(
                240,
                FEATURE_DIM,
                kernel_size=1,
                bias=False
            ),
            GN(FEATURE_DIM),
            nn.SiLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, pyramid):
        P1 = pyramid["P1"]
        P2 = self.p2_proj(pyramid["P2"])
        P3 = self.p3_proj(pyramid["P3"])
        P4 = self.p4_proj(pyramid["P4"])

        target_size = P2.shape[-3:]

        P3 = F.interpolate(
            P3,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )
        P4 = F.interpolate(
            P4,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )

        weights = self.weight_predictor(P2, P3, P4)

        w2 = weights[:, 0:1]
        w3 = weights[:, 1:2]
        w4 = weights[:, 2:3]

        shared = w2 * P2 + w3 * P3 + w4 * P4
        shared = self.refine(shared)

        high_resolution_feature = torch.cat(
            [
                P1,
                F.interpolate(
                    P2,
                    size=P1.shape[-3:],
                    mode="trilinear",
                    align_corners=False
                )
            ],
            dim=1
        )

        high_resolution_feature = self.recognition_projection(
            high_resolution_feature
        )

        global_feature = self.global_pool(shared).flatten(1)
        view_features = shared.permute(0, 2, 1, 3, 4).contiguous()

        return {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "P4": P4,
            "shared_feature": shared,
            "high_resolution_feature": high_resolution_feature,
            "view_features": view_features,
            "scale_weights": weights,
            "global_feature": global_feature
        }
