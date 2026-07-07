import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none"):
        super(IOUloss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        loss = 1.0 - iou * iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, strides=None):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides or [4]
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(self.strides)
        self.eps = 1e-6
        self.reg_max_log = 8.0

    def forward(self, inputs, labels=None):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            torch.cat(outputs, 1),
        )

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)
        xy = (output[..., :2] + grid.type_as(output)) * stride
        wh = torch.exp(output[..., 2:4].clamp(min=-self.reg_max_log, max=self.reg_max_log)) * stride
        decoded = torch.cat([xy, wh, output[..., 4:]], dim=-1)
        return decoded, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []
        num_fg = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4]
                self._validate_gt_classes(gt_classes, batch_idx)
                gt_classes = gt_classes.type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float()
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1.0)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum()
        loss_obj = self.bce_loss(obj_preds.view(-1, 1), obj_targets).sum()
        loss_cls = self.bce_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum()

        loss_iou = loss_iou / num_fg
        loss_obj = loss_obj / num_fg
        loss_cls = loss_cls / num_fg
        total_loss = 5.0 * loss_iou + loss_obj + loss_cls
        return {
            "loss": total_loss,
            "loss_iou": loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "num_fg": bbox_preds.new_tensor(float(num_fg)),
        }

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if num_in_boxes_anchor == 0:
            return gt_classes.new_zeros((0,)), fg_mask, gt_bboxes_per_image.new_zeros((0,)), gt_classes.new_zeros((0,), dtype=torch.long), 0

        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, xyxy=False)
        pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        pair_wise_ious_loss = -torch.log(pair_wise_ious.clamp_min(self.eps))

        cls_preds_ = torch.sigmoid(cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1))
        obj_preds_ = torch.sigmoid(obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1))
        cls_preds_ = cls_preds_ * obj_preds_
        cls_preds_ = torch.nan_to_num(cls_preds_, nan=0.0, posinf=1.0, neginf=0.0).clamp_(self.eps, 1.0 - self.eps)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt().clamp_(self.eps, 1.0 - self.eps), gt_cls_per_image, reduction="none").sum(-1)
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
            br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i + 1e-16)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius=2.5):
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        gt_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

        bbox_deltas = torch.stack([x_centers_per_image - gt_l, y_centers_per_image - gt_t, gt_r - x_centers_per_image, gt_b - y_centers_per_image], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_l = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_r = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_t = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_b = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        center_deltas = torch.stack([x_centers_per_image - gt_l, y_centers_per_image - gt_t, gt_r - x_centers_per_image, gt_b - y_centers_per_image], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def _validate_gt_classes(self, gt_classes, batch_idx):
        if gt_classes.numel() == 0:
            return
        gt_classes_long = gt_classes.to(torch.int64)
        if torch.any(gt_classes_long < 0) or torch.any(gt_classes_long >= self.num_classes):
            bad_values = gt_classes.detach().cpu().tolist()
            raise ValueError(
                "Invalid class ids in batch %d. Expected range [0, %d), got %s"
                % (batch_idx, self.num_classes, bad_values[:20])
            )
