import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from models.decoder import decoder
from models.pvtv2 import pvt_v2_b2


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def build_activation(name, inplace=False):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    if name == 'relu6':
        return nn.ReLU6(inplace=inplace)
    raise NotImplementedError(f'Unsupported activation: {name}')


def channel_mix_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class StageChannelRecalibrator(nn.Module):
    def __init__(self, channels, reduction=16, activation='relu'):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            build_activation(activation, inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        avg_response = self.mlp(self.avg_pool(x))
        max_response = self.mlp(self.max_pool(x))
        return self.gate(avg_response + max_response)


class StageSpatialRecalibrator(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        return self.gate(self.proj(torch.cat([avg_map, max_map], dim=1)))


class MultiKernelDepthwiseMixer(nn.Module):
    def __init__(self, channels, kernel_sizes, stride, activation='relu6', parallel=True):
        super().__init__()
        self.parallel = parallel
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                build_activation(activation, inplace=True),
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        running = x
        for branch in self.branches:
            branch_out = branch(running)
            outputs.append(branch_out)
            if not self.parallel:
                running = running + branch_out
        return outputs


class ResidualMultiScaleMixer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_sizes=(1, 3, 5),
        expansion=6,
        parallel=True,
        merge_add=True,
        activation='relu6',
    ):
        super().__init__()
        assert stride in [1, 2]
        expanded_channels = int(in_channels * expansion)
        self.use_skip = stride == 1
        self.merge_add = merge_add
        self.out_channels = out_channels
        self.expanded_channels = expanded_channels
        self.num_scales = len(kernel_sizes)

        self.channel_expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            build_activation(activation, inplace=True),
        )
        self.depthwise_mixer = MultiKernelDepthwiseMixer(
            expanded_channels,
            kernel_sizes,
            stride,
            activation,
            parallel=parallel,
        )
        merged_channels = expanded_channels if merge_add else expanded_channels * self.num_scales
        self.channel_project = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_proj = None
        if self.use_skip and in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        expanded = self.channel_expand(x)
        multi_scale_features = self.depthwise_mixer(expanded)
        if self.merge_add:
            mixed = 0
            for branch_out in multi_scale_features:
                mixed = mixed + branch_out
        else:
            mixed = torch.cat(multi_scale_features, dim=1)
        shuffle_groups = math.gcd(mixed.shape[1], self.out_channels)
        mixed = channel_mix_shuffle(mixed, shuffle_groups)
        projected = self.channel_project(mixed)
        if self.use_skip:
            shortcut = x if self.skip_proj is None else self.skip_proj(x)
            return shortcut + projected
        return projected


class ResidualMultiScaleEnhancer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=1,
        stride=1,
        kernel_sizes=(1, 3, 5),
        expansion=6,
        parallel=True,
        merge_add=True,
        activation='relu6',
    ):
        super().__init__()
        blocks = [
            ResidualMultiScaleMixer(
                in_channels,
                out_channels,
                stride,
                kernel_sizes=kernel_sizes,
                expansion=expansion,
                parallel=parallel,
                merge_add=merge_add,
                activation=activation,
            )
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualMultiScaleMixer(
                    out_channels,
                    out_channels,
                    1,
                    kernel_sizes=kernel_sizes,
                    expansion=expansion,
                    parallel=parallel,
                    merge_add=merge_add,
                    activation=activation,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SemanticLiftProjector(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.upsample_depthwise = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            build_activation(activation, inplace=True),
        )
        self.channel_project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.upsample_depthwise(x)
        x = channel_mix_shuffle(x, x.shape[1])
        return self.channel_project(x)


class GuidedLateralGate(nn.Module):
    def __init__(self, guide_channels, lateral_channels, hidden_channels, kernel_size=3, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        else:
            groups = max(1, hidden_channels)
        self.guide_proj = nn.Sequential(
            nn.Conv2d(
                guide_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(hidden_channels),
        )
        self.lateral_proj = nn.Sequential(
            nn.Conv2d(
                lateral_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(hidden_channels),
        )
        self.activation = build_activation(activation, inplace=True)
        self.mask_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, guide, lateral):
        guide_context = self.guide_proj(guide)
        lateral_context = self.lateral_proj(lateral)
        attention = self.mask_proj(self.activation(guide_context + lateral_context))
        return lateral * attention


class FRCR(nn.Module):
    def __init__(self, stage_channels, reference_dim=2048):
        super().__init__()
        self.frequency_projectors = nn.ModuleList(
            [nn.Conv2d(3, channels, 1) for channels in stage_channels]
        )
        self.reference_projectors = nn.ModuleList(
            [nn.Conv2d(reference_dim, channels, 1) for channels in stage_channels]
        )
        self.reference_sizes = [96, 48, 24, 12]

    def forward(self, query_image, stage_features, reference_feature):
        spectrum = torch.fft.rfft2(query_image, norm='ortho')
        magnitude = torch.abs(spectrum)
        resized_magnitude = F.interpolate(
            magnitude,
            size=stage_features[0].shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        stage_frequency_maps = [self.frequency_projectors[0](resized_magnitude)]
        for level in range(1, len(stage_features)):
            pooled_magnitude = F.avg_pool2d(resized_magnitude, 2 ** level)
            stage_frequency_maps.append(self.frequency_projectors[level](pooled_magnitude))

        if reference_feature.dim() == 2:
            reference_feature = reference_feature[:, :, None, None]
        elif reference_feature.dim() == 1:
            reference_feature = reference_feature[None, :, None, None]
        elif reference_feature.dim() == 5:
            reference_feature = reference_feature[:, 0, :, :, :]

        refined_features = []
        for level, feature in enumerate(stage_features):
            frequency_refined = feature * stage_frequency_maps[level]
            reference_map = self.reference_projectors[level](
                F.interpolate(
                    reference_feature,
                    size=(self.reference_sizes[level], self.reference_sizes[level]),
                    mode='bilinear',
                    align_corners=False,
                )
            )
            if level >= 2:
                refined_features.append(frequency_refined * (1 + reference_map))
            else:
                refined_features.append(frequency_refined * reference_map)

        return refined_features


class RFEB(nn.Module):
    def __init__(self, channels, use_attention):
        super().__init__()
        self.use_attention = use_attention
        self.channel_attention = StageChannelRecalibrator(channels) if use_attention else None
        self.spatial_attention = StageSpatialRecalibrator(3) if use_attention else None
        self.feature_enhancer = ResidualMultiScaleEnhancer(
            channels,
            channels,
            num_blocks=1,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion=6,
            parallel=True,
            merge_add=True,
            activation='relu6',
        )

    def forward(self, feature):
        enhanced_input = feature
        if self.use_attention:
            enhanced_input = enhanced_input * (1 + self.channel_attention(enhanced_input))
            enhanced_input = enhanced_input * (1 + self.spatial_attention(enhanced_input))
        return self.feature_enhancer(enhanced_input)


class SGFU(nn.Module):
    def __init__(self, deep_channels, lateral_channels):
        super().__init__()
        self.semantic_projection = SemanticLiftProjector(
            deep_channels,
            lateral_channels,
            kernel_size=3,
            activation='relu',
        )
        self.semantic_gate = GuidedLateralGate(
            guide_channels=lateral_channels,
            lateral_channels=lateral_channels,
            hidden_channels=lateral_channels // 2,
            kernel_size=3,
            activation='relu',
        )

    def forward(self, deeper_feature, lateral_feature):
        semantic_guidance = self.semantic_projection(deeper_feature)
        gated_lateral = self.semantic_gate(semantic_guidance, lateral_feature)
        return semantic_guidance + gated_lateral


class CRAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_projection = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.key_projection = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.current_value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.context_value_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.current_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.context_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.current_refine = nn.Sequential(
            ConvBNReLU(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(channels, channels, 1),
        )
        self.context_refine = nn.Sequential(
            ConvBNReLU(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(channels, channels, 1),
        )
        self.output_projection = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(channels, channels, 1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, current_feature, context_feature):
        query_seed = current_feature + context_feature
        key_seed = current_feature * context_feature

        batch_size, channels, height, width = query_seed.size()
        query = self.query_projection(query_seed).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_projection(key_seed).view(batch_size, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))

        current_value = self.current_value_projection(current_feature).view(batch_size, -1, width * height)
        context_value = self.context_value_projection(context_feature).view(batch_size, -1, width * height)

        refined_current = torch.bmm(current_value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        refined_context = torch.bmm(context_value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)

        refined_current = self.current_refine(self.current_scale * refined_current + current_feature)
        refined_context = self.context_refine(self.context_scale * refined_context + context_feature)
        return self.output_projection(refined_current + refined_context)


class PHKF(nn.Module):
    def __init__(self, stage_channels):
        super().__init__()
        c0, c1, c2, c3 = stage_channels

        self.rfeb_stage3 = RFEB(c3, use_attention=True)
        self.rfeb_stage2 = RFEB(c2, use_attention=True)
        self.rfeb_stage1 = RFEB(c1, use_attention=False)
        self.rfeb_stage0 = RFEB(c0, use_attention=False)

        self.sgfu_stage2 = SGFU(c3, c2)
        self.sgfu_stage1 = SGFU(c2, c1)
        self.sgfu_stage0 = SGFU(c1, c0)

        self.cram_stage3 = CRAM(c3)
        self.cram_stage2 = CRAM(c2)
        self.cram_stage1 = CRAM(c1)
        self.cram_stage0 = CRAM(c0)

        self.proj_3_to_2 = nn.Conv2d(c3, c2, kernel_size=1)
        self.proj_2_to_1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.proj_1_to_0 = nn.Conv2d(c1, c0, kernel_size=1)

    def forward(self, refined_features):
        x0, x1, x2, x3 = refined_features

        z3 = self.rfeb_stage3(x3)
        y2 = self.sgfu_stage2(z3, x2)
        z2 = self.rfeb_stage2(y2)
        y1 = self.sgfu_stage1(z2, x1)
        z1 = self.rfeb_stage1(y1)
        y0 = self.sgfu_stage0(z1, x0)
        z0 = self.rfeb_stage0(y0)

        out3 = self.cram_stage3(z3, z3)
        out3_to_2 = self.proj_3_to_2(F.interpolate(out3, size=z2.shape[2:], mode='bilinear', align_corners=False))
        out2 = self.cram_stage2(z2, out3_to_2)
        out2_to_1 = self.proj_2_to_1(F.interpolate(out2, size=z1.shape[2:], mode='bilinear', align_corners=False))
        out1 = self.cram_stage1(z1, out2_to_1)
        out1_to_0 = self.proj_1_to_0(F.interpolate(out1, size=z0.shape[2:], mode='bilinear', align_corners=False))
        out0 = self.cram_stage0(z0, out1_to_0)

        return out0, out1, out2, out3


class MultiScalePredictionHead(nn.Module):
    def __init__(self, stage_channels, decoder_channels):
        super().__init__()
        self.decoder = decoder(dims=stage_channels, nclass=decoder_channels)
        self.logit_projection = nn.Conv2d(decoder_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, aggregated_features):
        out0, out1, out2, out3 = aggregated_features
        s0, s1, s2, s3 = self.decoder(out3, out2, out1, out0)
        s3 = F.interpolate(s3, scale_factor=16, mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, scale_factor=8, mode='bilinear', align_corners=False)
        s1 = F.interpolate(s1, scale_factor=4, mode='bilinear', align_corners=False)
        return (
            self.logit_projection(s3),
            self.logit_projection(s2),
            self.logit_projection(s1),
            self.logit_projection(s0),
        )


class FHKMNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = pvt_v2_b2()
        pretrained_path = getattr(opt, 'pvt_weights', './pvt_weights/pvt_v2_b2.pth')
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f'PVTv2 pretrained weights not found: {pretrained_path}. '
                'Please download pvt_v2_b2.pth into ./pvt_weights or pass --pvt_weights explicitly.'
            )
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        backbone_state = self.backbone.state_dict()
        compatible_state = {k: v for k, v in pretrained_state.items() if k in backbone_state}
        backbone_state.update(compatible_state)
        self.backbone.load_state_dict(backbone_state)

        self.stage_channels = [64, 128, 320, 512]
        self.FRCR = FRCR(self.stage_channels, reference_dim=2048)
        self.PHKF = PHKF(self.stage_channels)
        self.prediction_head = MultiScalePredictionHead(self.stage_channels, decoder_channels=opt.dim)

    def forward(self, x, ref_x, y=None, training=True, un_list=None):
        stage_features = self.backbone(x)
        refined_features = self.FRCR(
            query_image=x,
            stage_features=stage_features,
            reference_feature=ref_x,
        )
        aggregated_features = self.PHKF(refined_features)
        return self.prediction_head(aggregated_features)
