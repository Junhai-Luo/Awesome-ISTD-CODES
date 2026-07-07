import torch
import torch.nn as nn

from .DQAligner import DQAligner


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleHeadInput(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.down = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3, 2, 1),
            ConvBNAct(hidden_channels, hidden_channels, 3, 2, 1),
            ConvBNAct(hidden_channels, hidden_channels, 3, 2, 1),
        )

    def forward(self, x):
        return self.down(x)


class YOLOXHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, hidden_channels=128):
        super().__init__()
        self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cls_convs = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.reg_convs = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(hidden_channels, 4, 1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class DQAlignerBaseDet(nn.Module):
    def __init__(self, input_channels=3, num_frames=5, num_classes=1, key_mode="last"):
        super().__init__()
        self.backbone = DQAligner(
            input_channels=input_channels,
            num_frames=num_frames,
            train_mode=False,
            key_mode=key_mode,
        )
        self.num_frames = int(num_frames)
        self.key_mode = key_mode
        self.num_classes = int(num_classes)

    def _forward_backbone_features(self, x):
        out_l1, out_l2, out_l3, _ = self.backbone.feat_extract(x)
        out_l1, out_l2, out_l3 = self.backbone.multi_att([out_l1, out_l2, out_l3])
        _, obj_query = self.backbone.object_query(out_l1)

        if self.key_mode == "mid":
            key_index = out_l1.shape[2] // 2
        else:
            key_index = out_l1.shape[2] - 1

        key_feat = [
            out_l1[:, :, key_index, :, :],
            out_l2[:, :, key_index, :, :],
            out_l3[:, :, key_index, :, :],
        ]
        align_feat_frame = []
        key_feat_out = None
        for t in range(out_l1.shape[2]):
            if t == key_index:
                key_feat_out, _ = self.backbone.key_enhance_l1(key_feat[0], obj_query[:, t])
                continue
            ref_feat = [
                out_l1[:, :, t, :, :],
                out_l2[:, :, t, :, :],
                out_l3[:, :, t, :, :],
            ]
            align_feat, _ = self.backbone.mask_align(ref_feat, key_feat, obj_query[:, t])
            align_feat_frame.append(align_feat)

        if key_feat_out is None:
            key_feat_out = key_feat[0]
        align_feat_frame = torch.stack(align_feat_frame, dim=2)
        align_feat_frame = torch.cat([align_feat_frame, key_feat_out.unsqueeze(2)], dim=2)
        align_feat_frame = self.backbone.temporal_attn(align_feat_frame)
        mask_logits = self.backbone.output3d_0(align_feat_frame).squeeze(2)
        return align_feat_frame, mask_logits


class DQAlignerDet(DQAlignerBaseDet):
    """Feature-detection variant: aligned temporal feature -> detection head."""

    def __init__(self, input_channels=3, num_frames=5, num_classes=1, key_mode="last"):
        super().__init__(input_channels=input_channels, num_frames=num_frames, num_classes=num_classes, key_mode=key_mode)
        self.downsample = DownsampleHeadInput(16, hidden_channels=64)
        self.head = YOLOXHead(64, num_classes=num_classes)

    def forward(self, x):
        align_feat_frame, _ = self._forward_backbone_features(x)
        feat = align_feat_frame.mean(dim=2)
        feat = self.downsample(feat)
        return [self.head(feat)]


class DQAlignerSaliencyDet(DQAlignerBaseDet):
    """Saliency-detection variant: mask logits -> detection head."""

    def __init__(self, input_channels=3, num_frames=5, num_classes=1, key_mode="last"):
        super().__init__(input_channels=input_channels, num_frames=num_frames, num_classes=num_classes, key_mode=key_mode)
        self.downsample = DownsampleHeadInput(1, hidden_channels=32)
        self.head = YOLOXHead(32, num_classes=num_classes)

    def forward(self, x):
        _, mask_logits = self._forward_backbone_features(x)
        feat = self.downsample(mask_logits)
        return [self.head(feat)]
