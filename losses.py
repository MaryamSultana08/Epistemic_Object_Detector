import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retinanet import random_set

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


def kl_dirichlet(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = torch.clamp(p, min=eps, max=1e4)
    q = torch.clamp(q, min=eps, max=1e4)

    p64 = p.double()
    q64 = q.double()

    p_sum = torch.clamp(p64.sum(dim=-1), min=eps, max=1e4)
    q_sum = torch.clamp(q64.sum(dim=-1), min=eps, max=1e4)

    term1 = torch.lgamma(p_sum) - torch.lgamma(q_sum)
    term2 = torch.sum(torch.lgamma(q64) - torch.lgamma(p64), dim=-1)

    dig_p = torch.digamma(p64)
    dig_p_sum = torch.digamma(p_sum).unsqueeze(-1)
    term3 = torch.sum((p64 - q64) * (dig_p - dig_p_sum), dim=-1)

    out = term1 + term2 + term3
    out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)
    out = torch.clamp(out, 0.0, 1e6)

    return out.float()


class DirichletFocalLoss(nn.Module):
    def __init__(
        self,
        use_random_set=False,
        use_random_set_betp_loss=False,
        random_set_path=None,
        random_set_alpha=0.001,
        random_set_beta=0.001,
        random_set_base_class_names=None,
        coord_l1_weight=1.0,
        kl_weight=0.005,
        delta_clip=3.0,
        target_concentration=20.0,
    ):
        super().__init__()
        self.use_random_set = bool(use_random_set)
        self.use_random_set_betp_loss = bool(use_random_set_betp_loss)
        self.random_set_alpha = float(random_set_alpha)
        self.random_set_beta = float(random_set_beta)
        self.coord_l1_weight = float(coord_l1_weight)
        self.kl_weight = float(kl_weight)
        self.delta_clip = float(delta_clip)
        self.target_concentration = float(target_concentration)
        self.target_eps = 1e-3

        if self.use_random_set:
            if not random_set_path:
                raise ValueError("random_set_path is required when use_random_set=True.")
            if random_set_base_class_names is None:
                base_class_names = [c["name"] for c in random_set.COCO_CATEGORIES]
            else:
                base_class_names = [str(x) for x in random_set_base_class_names]
            new_classes = random_set.load_random_set_classes(random_set_path)
            mats = random_set.build_random_set_matrices(new_classes, base_class_names)
            self.rs_membership = mats["membership"]
            self.rs_mass_coeff = mats["mass_coeff"]
            self.rs_pignistic = mats["pignistic"]

    def _alphas_to_norm(self, alphas: torch.Tensor) -> torch.Tensor:
        a = alphas.view(-1, 4, 3)
        a_sum = torch.clamp(a.sum(dim=-1, keepdim=True), min=1e-6)
        p_mean = a / a_sum
        p_mean = torch.nan_to_num(p_mean, nan=0.0, posinf=1.0, neginf=0.0)
        bin_centers = torch.tensor([0.1667, 0.5, 0.8333], device=alphas.device)
        coords01 = (p_mean * bin_centers.view(1, 1, 3)).sum(dim=-1)
        return torch.clamp(coords01, 0.0, 1.0)

    def _norm_to_deltas(self, norm: torch.Tensor) -> torch.Tensor:
        return (norm - 0.5) * (2.0 * self.delta_clip)

    def _deltas_to_norm(self, deltas: torch.Tensor) -> torch.Tensor:
        deltas = torch.clamp(deltas, -self.delta_clip, self.delta_clip)
        return deltas / (2.0 * self.delta_clip) + 0.5

    def _norm_to_target_alphas(self, norm: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(norm, 0.0, 1.0)
        bins = torch.tensor([0.1667, 0.5, 0.8333], device=norm.device)
        idx = torch.bucketize(t, bins)
        lower = torch.clamp(idx - 1, 0, len(bins) - 1)
        upper = torch.clamp(idx, 0, len(bins) - 1)

        lower_bins = bins[lower]
        upper_bins = bins[upper]
        denom = torch.where(upper_bins > lower_bins, upper_bins - lower_bins, torch.ones_like(upper_bins))
        w_upper = torch.where(upper_bins > lower_bins, (t - lower_bins) / denom, torch.zeros_like(t))
        w_lower = 1.0 - w_upper

        probs = torch.zeros(t.shape + (len(bins),), device=t.device)
        probs.scatter_(-1, lower.unsqueeze(-1), w_lower.unsqueeze(-1))
        probs.scatter_add_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))

        alphas = probs * self.target_concentration + self.target_eps
        return alphas

    def _binary_cross_entropy_rs(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        y_true = torch.clamp(y_true, eps, 1.0)
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        term_0 = (1.0 - y_true) * torch.log(1.0 - y_pred + eps)
        term_1 = y_true * torch.log(y_pred + eps)
        return -(term_0 + term_1).mean(dim=1)

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        kl_losses = []
        l1_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if self.use_random_set:
                    if self.use_random_set_betp_loss:
                        rs_mass_coeff = self.rs_mass_coeff.to(classification.device)
                        rs_pignistic = self.rs_pignistic.to(classification.device)
                        mass = random_set.belief_to_mass(classification, rs_mass_coeff, clamp_negative=True)
                        betp = random_set.final_betp(mass, rs_pignistic)
                        betp = torch.clamp(betp, 1e-4, 1.0 - 1e-4)

                        alpha_factor = torch.ones(betp.shape, device=classification.device) * alpha
                        alpha_factor = 1.0 - alpha_factor
                        focal_weight = betp
                        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                        bce = -(torch.log(1.0 - betp))
                        cls_loss = focal_weight * bce
                        # Normalize by number of entries (anchors x classes) to keep
                        # empty-GT BetP loss scale comparable across datasets.
                        num_entries = torch.clamp(
                            torch.tensor(float(cls_loss.numel()), device=classification.device),
                            min=1.0,
                        )
                        classification_losses.append(cls_loss.sum() / num_entries)
                    else:
                        gt_bel = torch.zeros_like(classification)
                        bce = self._binary_cross_entropy_rs(gt_bel, classification)
                        bce_loss = bce.sum()

                        rs_mass_coeff = self.rs_mass_coeff.to(classification.device)
                        mass = random_set.belief_to_mass(classification, rs_mass_coeff, clamp_negative=False)
                        mass_reg = F.relu(-mass).mean()
                        mass_sum = F.relu(mass.sum(dim=-1).mean() - 1.0)
                        cls_loss = bce_loss + self.random_set_alpha * mass_reg + self.random_set_beta * mass_sum
                        classification_losses.append(cls_loss)
                else:
                    alpha_factor = torch.ones(classification.shape, device=classification.device) * alpha
                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    bce = -(torch.log(1.0 - classification))
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())

                regression_losses.append(torch.tensor(0.0, device=classification.device))
                kl_losses.append(torch.tensor(0.0, device=classification.device))
                l1_losses.append(torch.tensor(0.0, device=classification.device))
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            if self.use_random_set:
                if self.use_random_set_betp_loss:
                    rs_mass_coeff = self.rs_mass_coeff.to(classification.device)
                    rs_pignistic = self.rs_pignistic.to(classification.device)
                    mass = random_set.belief_to_mass(classification, rs_mass_coeff, clamp_negative=True)
                    betp = random_set.final_betp(mass, rs_pignistic)
                    betp = torch.clamp(betp, 1e-4, 1.0 - 1e-4)

                    targets = torch.ones(betp.shape, device=classification.device) * -1
                    targets[torch.lt(IoU_max, 0.4), :] = 0
                    targets[positive_indices, :] = 0

                    if num_positive_anchors > 0:
                        label_ids = assigned_annotations[positive_indices, 4].long()
                        valid = (label_ids >= 0) & (label_ids < betp.shape[1])
                        if valid.any():
                            pos_idx = torch.where(positive_indices)[0]
                            targets[pos_idx[valid], label_ids[valid]] = 1

                    alpha_factor = torch.ones(targets.shape, device=classification.device) * alpha
                    alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
                    focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - betp, betp)
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(targets * torch.log(betp) + (1.0 - targets) * torch.log(1.0 - betp))
                    cls_loss = focal_weight * bce
                    valid_mask = torch.ne(targets, -1.0)
                    cls_loss = torch.where(
                        valid_mask,
                        cls_loss,
                        torch.zeros(cls_loss.shape, device=classification.device),
                    )

                    # BetP negatives on ROAD can dominate when normalized only by
                    # positive anchors. Normalize positive and negative terms
                    # separately for stable training.
                    pos_mask = torch.eq(targets, 1.0)
                    neg_mask = torch.eq(targets, 0.0)

                    if pos_mask.any():
                        pos_count = torch.clamp(pos_mask.sum().float(), min=1.0)
                        pos_loss = cls_loss[pos_mask].sum() / pos_count
                    else:
                        pos_loss = torch.tensor(0.0, device=classification.device)

                    if neg_mask.any():
                        neg_count = torch.clamp(neg_mask.sum().float(), min=1.0)
                        neg_loss = cls_loss[neg_mask].sum() / neg_count
                    else:
                        neg_loss = torch.tensor(0.0, device=classification.device)

                    classification_losses.append(pos_loss + neg_loss)
                else:
                    gt_bel_target = torch.zeros_like(classification)
                    if num_positive_anchors > 0:
                        label_ids = assigned_annotations[positive_indices, 4].long()
                        rs_membership = self.rs_membership.to(classification.device)
                        valid = (label_ids >= 0) & (label_ids < rs_membership.shape[0])
                        if valid.any():
                            pos_idx = torch.where(positive_indices)[0]
                            gt_bel_target[pos_idx[valid]] = rs_membership[label_ids[valid]]

                    valid_indices = torch.lt(IoU_max, 0.4) | positive_indices
                    bel_pred_valid = classification[valid_indices]
                    gt_bel_valid = gt_bel_target[valid_indices]
                    if bel_pred_valid.numel() == 0:
                        classification_losses.append(torch.tensor(0.0, device=classification.device))
                    else:
                        bce = self._binary_cross_entropy_rs(gt_bel_valid, bel_pred_valid)
                        bce_loss = bce.sum() / max(1, num_positive_anchors.item())

                        rs_mass_coeff = self.rs_mass_coeff.to(classification.device)
                        mass = random_set.belief_to_mass(bel_pred_valid, rs_mass_coeff, clamp_negative=False)
                        mass_reg = F.relu(-mass).mean()
                        mass_sum = F.relu(mass.sum(dim=-1).mean() - 1.0)

                        cls_loss = bce_loss + self.random_set_alpha * mass_reg + self.random_set_beta * mass_sum
                        classification_losses.append(cls_loss)
            else:
                targets = torch.ones(classification.shape, device=classification.device) * -1
                targets[torch.lt(IoU_max, 0.4), :] = 0
                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

                alpha_factor = torch.ones(targets.shape, device=classification.device) * alpha
                alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=classification.device))

                classification_losses.append(
                    cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
                )

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).t()
                targets = targets / torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=targets.device)

                pred_alphas = regression[positive_indices, :]
                pred_alphas = torch.clamp(pred_alphas, min=1e-3, max=20.0)

                target_regs = torch.clamp(targets, -self.delta_clip, self.delta_clip)
                target_norm = self._deltas_to_norm(target_regs)
                gt_alphas = self._norm_to_target_alphas(target_norm)

                gt_a = gt_alphas.view(-1, 3)
                pr_a = pred_alphas.view(-1, 3)
                kl = kl_dirichlet(pr_a, gt_a)
                kl = kl.view(-1, 4).sum(dim=-1)
                kl_loss = torch.clamp(kl.mean(), 0.0, 1e6)

                pred_norm = self._alphas_to_norm(pred_alphas)
                pred_deltas = self._norm_to_deltas(pred_norm)
                pred_deltas = torch.nan_to_num(pred_deltas, nan=0.0, posinf=0.0, neginf=0.0)

                l1 = F.smooth_l1_loss(pred_deltas, target_regs, reduction="none", beta=1.0 / 9.0).sum(dim=-1)
                l1_loss = l1.mean()

                reg_loss = self.kl_weight * kl_loss + self.coord_l1_weight * l1_loss
                regression_losses.append(reg_loss)
                kl_losses.append(kl_loss)
                l1_losses.append(l1_loss)
            else:
                regression_losses.append(torch.tensor(0.0, device=classification.device))
                kl_losses.append(torch.tensor(0.0, device=classification.device))
                l1_losses.append(torch.tensor(0.0, device=classification.device))

        mean_cls = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        mean_reg = torch.stack(regression_losses).mean(dim=0, keepdim=True)
        mean_kl = torch.stack(kl_losses).mean(dim=0, keepdim=True)
        mean_l1 = torch.stack(l1_losses).mean(dim=0, keepdim=True)

        self.last_kl = mean_kl
        self.last_l1 = mean_l1

        return mean_cls, mean_reg, mean_kl, mean_l1
