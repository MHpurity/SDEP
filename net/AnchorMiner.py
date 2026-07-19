import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMConv2d(nn.Module):
    def __init__(self, stride: int = 1, padding: int = 2, clip: float = 2.0):
        super().__init__()
        self.clip = clip
        filters = self._build_filters()
        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=stride, padding=padding, bias=False)
        self.conv.weight = nn.Parameter(filters, requires_grad=False)

    def _build_filters(self) -> torch.Tensor:
        f1 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        f2 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        f3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0
        filters = np.stack([
            np.stack([f1, f1, f1], axis=0),
            np.stack([f2, f2, f2], axis=0),
            np.stack([f3, f3, f3], axis=0),
        ], axis=0)
        return torch.from_numpy(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.clip is not None and self.clip > 0:
            y = torch.clamp(y, -self.clip, self.clip)
        return y


class AnchorMiner(nn.Module):
    """Unchanged phase-1 miner. Used only for training supervision / diagnostics."""
    def __init__(self, top_k=256, clamp_val=2.0):
        super().__init__()
        self.srm = SRMConv2d(stride=1, padding=2, clip=0.0)
        self.clamp_val = clamp_val
        self.top_k = top_k
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images, sam_masks):
        B, C, H, W = images.shape
        r = self.srm(images)
        r = torch.clamp(r, -self.clamp_val, self.clamp_val)

        mu_3 = F.avg_pool2d(r, kernel_size=3, stride=1, padding=1)
        e_3 = F.avg_pool2d(r ** 2, kernel_size=3, stride=1, padding=1)
        var_3 = e_3 - mu_3 ** 2

        mu_5 = F.avg_pool2d(r, kernel_size=5, stride=1, padding=2)
        e_5 = F.avg_pool2d(r ** 2, kernel_size=5, stride=1, padding=2)
        var_5 = e_5 - mu_5 ** 2

        f_local = torch.cat([var_3, var_5], dim=1)

        dilated_masks = F.max_pool2d(sam_masks, kernel_size=15, stride=1, padding=7)
        shell_masks = torch.clamp(dilated_masks - sam_masks, 0.0, 1.0)

        shell_sum = shell_masks.sum(dim=(2, 3), keepdim=True) + 1e-6
        p_shell = (f_local * shell_masks).sum(dim=(2, 3), keepdim=True) / shell_sum

        sim = F.cosine_similarity(f_local, p_shell, dim=1)
        anomaly_map = 1.0 - sim

        sam_masks_squeeze = sam_masks.squeeze(1)
        anchor_weight = sam_masks_squeeze * anomaly_map

        K = min(self.top_k, H * W)
        rank = torch.arange(K, device=images.device).unsqueeze(0).expand(B, K)

        # Positive anchors
        pos_candidate = (sam_masks_squeeze > 0.5)  # [B,H,W]
        pos_candidate_flat = pos_candidate.view(B, -1)
        pos_count = pos_candidate_flat.sum(dim=1)

        pos_weight_flat = anchor_weight.view(B, -1)
        pos_weight_masked = pos_weight_flat.masked_fill(~pos_candidate_flat, -1e6)
        _, pos_topk_indices = torch.topk(pos_weight_masked, K, dim=1)

        pos_valid_count = pos_count.clamp(max=K)
        pos_valid = rank < pos_valid_count.unsqueeze(1) 
        pos_y = pos_topk_indices // W
        pos_x = pos_topk_indices % W


        # Negative anchors
        bg_masks_flat = (1.0 - dilated_masks.squeeze(1)).view(B, -1)
        bg_rand = torch.rand_like(bg_masks_flat) * bg_masks_flat
        _, bg_topk_indices = torch.topk(bg_rand, K, dim=1)
        neg_y = bg_topk_indices // W
        neg_x = bg_topk_indices % W

        return (pos_y, pos_x, pos_valid), (neg_y, neg_x), anomaly_map.unsqueeze(1), anchor_weight.unsqueeze(1)