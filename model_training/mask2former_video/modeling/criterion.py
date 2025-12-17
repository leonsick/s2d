# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from .point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.debugging import *


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        weights: Optional[torch.Tensor] = None,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    if weights is not None:
        # If weights are provided, apply them to the loss
        weights = weights.mean(-1)
        assert weights.shape == loss.shape, f"Weights must have the same shape as loss but got loss {loss.shape} and weights {weights.shape}"
        loss = loss * weights

    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        weights: Optional[torch.Tensor] = None,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if weights is not None:
        # If weights are provided, apply them to the loss
        assert weights.shape == loss.shape, f"Weights must have the same shape as loss but got loss {loss.shape} and weights {weights.shape}"
        loss = loss * weights

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def dice_loss_weight(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        loss_weight: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss * loss_weight
    return loss.sum() / num_masks


dice_loss_weight_jit = torch.jit.script(dice_loss_weight)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss_weight(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        loss_weight: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if loss_weight is not None:
        # If weights are provided, apply them to the loss
        # assert loss_weight.shape == loss.shape, f"Weights must have the same shape as loss but got loss {loss.shape} and weights {weights.shape}"
        # print("loss_weight.shape", loss_weight.shape)
        print("loss.mean(1).shape", loss.mean(1).shape)

        loss = loss.mean(1) * loss_weight

    return loss.sum() / num_masks


sigmoid_ce_loss_weight_jit = torch.jit.script(sigmoid_ce_loss_weight)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, loss_strategy, reweight_distillation_loss=False,
                 distillation_loss_strategy="masks-only"):

        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.loss_strategy = loss_strategy
        self.reweight_distillation_loss = reweight_distillation_loss
        self.distillation_loss_strategy = distillation_loss_strategy

        assert self.loss_strategy in ["masks-only", "full"], \
            f"Unknown loss strategy {self.loss_strategy}. Choose from ['masks-only', 'full']"

    def loss_entropy(self, outputs, targets, indices, num_masks, distillation):
        """
        Entropy loss to encourage every pixel to only belong to one mask.
        This loss is calculated over all predicted masks.
        """
        # print("INSIDE ENTROPY")
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"]  # Shape: (B, Q, T, H, W)

        logits = src_masks.flatten(2)  # Shape: (B, Q, T*H*W)
        probabilities = F.softmax(logits, dim=1)

        # Add a small epsilon for numerical stability
        epsilon = 1e-6
        probabilities = probabilities.clamp(min=epsilon)

        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)

        loss = entropy.mean()

        losses = {"loss_entropy": loss}
        return losses

    def loss_labels(self, outputs, targets, indices, num_masks, distillation):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """



        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes_o = torch.zeros_like(target_classes_o)

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_labels_sparse(self, outputs, targets, indices, num_masks, distillation):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        # TODO: Calculate mask loss only from frames with at least one GT mask
        # print("INSIDE LOSS LABELS SPARSE")

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # TO FIX THE CLASS BUG: If class is 1, set to 0, if class is 0 set to 1
        target_classes_o = torch.zeros_like(target_classes_o)

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # Scores for all predictions except the matched ones
        all_scores = F.softmax(src_logits, dim=-1)[:, :, :-1].squeeze(-1)  # (N, Q, C+1)

        all_scores[idx] = 1.0  # Set the scores of the matched predictions to 1.0 (to exclude them)

        topk_lowest_scores, topk_ls_indices = torch.topk(all_scores, k=10, dim=-1, largest=False)
        lowest_score_indicies = (torch.arange(topk_ls_indices.shape[0]).repeat_interleave(topk_ls_indices.shape[1]),
                                 topk_ls_indices.flatten())


        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")

        loss_ce = torch.cat((loss_ce[idx], loss_ce[lowest_score_indicies]), dim=0).mean()

        losses = {"loss_ce_sparse": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, distillation):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        # Temporal DropLoss: Remove masks with zero area
        if (self.loss_strategy == "masks-only" and not distillation) or (
                self.distillation_loss_strategy == "masks-only" and distillation):
            keep_idx = []
            for i_msk, msk in enumerate(target_masks):
                if not msk.sum() == 0:
                    keep_idx.append(i_msk)

            if len(keep_idx) == 0:
                logging.warning("No target masks with non-zero area found! Returning empty losses.")
                return {"loss_mask": torch.tensor(0.0, device=src_masks.device),
                        "loss_dice": torch.tensor(0.0, device=src_masks.device)}

            keep_idx = torch.tensor(keep_idx, dtype=torch.int64, device=src_masks.device)
            target_masks = target_masks[keep_idx]
            src_masks = src_masks[keep_idx]

        with torch.no_grad():
            # sample point_coords

            # This is the original method to sample point coordinates
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, None),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks, None),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, distillation=False):
        """loss_map = {
            'labels': self.loss_labels,
            'labels_drop': self.loss_labels_drop,
            'masks': self.loss_masks,
            'masks_drop': self.loss_masks_drop,
            'labels_sparse': self.loss_labels_sparse,
            'entropy': self.loss_entropy,
        }"""

        loss_map = {
            'labels': self.loss_labels,
            'labels_drop': self.loss_labels_drop,
            'masks': self.loss_masks,
            'masks_drop': self.loss_masks_drop,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, distillation)

    def forward(self, outputs, targets, distillation=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, distillation=distillation))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "labels":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
