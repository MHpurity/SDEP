import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TinyViT.models.tiny_vit import tiny_vit_5m_224


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
                      padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class TinyViTBackboneAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def get_intermediate_features(self, x):
        x0 = self.backbone.patch_embed(x)  # (B, 64, 56, 56)
        H0, W0 = x0.shape[2], x0.shape[3]
        f0 = x0.flatten(2).transpose(1, 2).contiguous()

        features = [f0]
        resolutions = [(H0, W0)]

        x1 = self.backbone.layers[0](x0)
        H1, W1 = H0 // 2, W0 // 2
        features.append(x1)
        resolutions.append((H1, W1))

        x2 = self.backbone.layers[1](x1)
        H2, W2 = H1 // 2, W1 // 2
        features.append(x2)
        resolutions.append((H2, W2))

        x3 = self.backbone.layers[2](x2)
        H3, W3 = H2 // 2, W2 // 2
        features.append(x3)
        resolutions.append((H3, W3))

        x4 = self.backbone.layers[3](x3)
        features.append(x4)
        resolutions.append((H3, W3))

        return features, resolutions


class SharedFPH(nn.Module):
    def __init__(self, in_channels=64, proj_channels=128):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNR(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels)
        )
    def forward(self, x):
        return self.block(x)


class BoundaryRefineBlock(nn.Module):
    def __init__(self, low_channels=32, high_channels=64, out_channels=64):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNR(low_channels + high_channels, out_channels, 3),
            ConvBNR(out_channels, out_channels, 3)
        )

    def forward(self, low_feat, high_feat):
        if high_feat.size()[2:] != low_feat.size()[2:]:
            high_feat = F.interpolate(high_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([low_feat, high_feat], dim=1))


class AnchorIML(nn.Module):
    def __init__(self, evi_lambda: float = 0.5, residual_beta: float = 0.3, alpha_temp: float = 1.0):
        super().__init__()
        self.backbone = TinyViTBackboneAdapter(tiny_vit_5m_224(pretrained=True))
        self.evi_lambda = evi_lambda
        self.residual_beta = residual_beta
        self.alpha_temp = alpha_temp

        self.conv1x1_0 = Conv1x1(64, 32)
        self.conv1x1_1 = Conv1x1(128, 64)
        self.conv1x1_2 = Conv1x1(160, 64)
        self.conv1x1_3 = Conv1x1(320, 64)
        self.conv1x1_4 = Conv1x1(320, 64)

        self.fph = SharedFPH(in_channels=64, proj_channels=128)
        self.proto_forged = nn.Parameter(torch.randn(1, 128, 1, 1))
        self.proto_pristine = nn.Parameter(torch.randn(1, 128, 1, 1))

        nn.init.normal_(self.proto_forged, std=0.01)
        nn.init.normal_(self.proto_pristine, std=0.01)
        self.tau = 0.07

        # MLP
        self.scale_mlp = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1)
        )

        self.base_fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.evi_fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_fuse = nn.Sequential(
            ConvBNR(64, 64, 3),
            ConvBNR(64, 64, 3)
        )
        self.boundary_refine = BoundaryRefineBlock(low_channels=32, high_channels=64, out_channels=64)
        self.last_conv1x1 = nn.Conv2d(64, 1, 1, bias=True)

    @staticmethod
    def _token_to_map(tokens, B, H, W):
        return tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, return_dict=False):
        image_shape = x.shape[2:]
        B = x.shape[0]
        features, resolutions = self.backbone.get_intermediate_features(x)

        x0 = self.conv1x1_0(self._token_to_map(features[0], B, resolutions[0][0], resolutions[0][1]))
        x1 = self.conv1x1_1(self._token_to_map(features[1], B, resolutions[1][0], resolutions[1][1]))
        x2 = self.conv1x1_2(self._token_to_map(features[2], B, resolutions[2][0], resolutions[2][1]))
        x3 = self.conv1x1_3(self._token_to_map(features[3], B, resolutions[3][0], resolutions[3][1]))
        x4 = self.conv1x1_4(self._token_to_map(features[4], B, resolutions[4][0], resolutions[4][1]))

        target_size = x1.shape[2:]
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)

        feats = [x1, x2, x3, x4]

        e_logits = []
        e_probs = []
        modulated_feats = []
        scale_scores = []
        sim_f_list = []
        sim_p_list = []

        p_f = F.normalize(self.proto_forged, p=2, dim=1)
        p_p = F.normalize(self.proto_pristine, p=2, dim=1)

        for feat in feats:
            f_phys = self.fph(feat)   #F_i
            f_phys_norm = F.normalize(f_phys, p=2, dim=1)

            sim_f = (f_phys_norm * p_f).sum(dim=1, keepdim=True)
            sim_p = (f_phys_norm * p_p).sum(dim=1, keepdim=True)
            sim_f_list.append(sim_f)
            sim_p_list.append(sim_p)
            e_logit = (sim_f - sim_p) / self.tau  # G_i
            e_prob = torch.sigmoid(e_logit)

            e_logits.append(e_logit)
            e_probs.append(e_prob)


            modulated_feats.append(feat * (1.0 + self.evi_lambda * e_prob))
            feat_vec = F.adaptive_avg_pool2d(f_phys, output_size=1).flatten(1)
            scale_score = self.scale_mlp(feat_vec)
            scale_scores.append(scale_score)

        score_mat = torch.cat(scale_scores, dim=1) / self.alpha_temp
        alpha = F.softmax(score_mat, dim=1)

        f_base_pre = feats[0] + feats[1] + feats[2] + feats[3]
        f_base = self.base_fusion_conv(f_base_pre)

        alpha_view = alpha.view(B, 4, 1, 1, 1)
        stacked_modulated = torch.stack(modulated_feats, dim=1)  # [B,4,C,H,W]
        f_evi_pre = (alpha_view * stacked_modulated).sum(dim=1)
        f_evi = self.evi_fusion_conv(f_evi_pre)

        fused = self.final_fuse(f_base + self.residual_beta * f_evi)
        boundary_feat = self.boundary_refine(x0, fused)
        logits = self.last_conv1x1(boundary_feat)
        logits = F.interpolate(logits, size=image_shape, mode='bilinear', align_corners=False)

        if not return_dict:
            return logits

        alpha_maps = [alpha[:, i].view(B, 1, 1, 1) for i in range(4)]
        weighted_evidence = sum(alpha_maps[i] * e_probs[i] for i in range(4))

        proto_sim = F.cosine_similarity(p_f, p_p, dim=1).squeeze()

        return {
            'logits': logits,
            'e_logits': e_logits
        }
