# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import math
import os
import random
import warnings
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom
from .data_video.dataset_mapper import apply_transformation_frame_by_frame, apply_transformslist_frame_by_frame

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class KDVideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            student_backbone: Backbone,
            student_sem_seg_head: nn.Module,
            teacher_backbone: Backbone,
            teacher_sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # video
            num_frames,
            num_predictions_distillation,
            score_threshold_distillation,
            use_nms: bool = False,
            nms_threshold: float = 0.75,
            num_predictions_inference: int = 10,
            distillation_nms: bool = False,
            accum_iter: int = 1,  # Number of iterations to accumulate gradients before updating the model
            disentangle_distillation_loader: bool = False,
            eval_student: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        # self.student_backbone = student_backbone
        # self.teacher_backbone = teacher_backbone

        # self.student_sem_seg_head = student_sem_seg_head
        # self.teacher_sem_seg_head = teacher_sem_seg_head
        self.student = nn.Sequential(student_backbone, student_sem_seg_head)
        self.teacher = nn.Sequential(teacher_backbone, teacher_sem_seg_head)

        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_predictions_distillation = num_predictions_distillation
        self.score_threshold_distillation = score_threshold_distillation
        self.num_predictions_inference = num_predictions_inference

        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.num_predictions_inference = num_predictions_inference

        self.distillation_nms = distillation_nms

        self.accum_iter = accum_iter  # Number of iterations to accumulate gradients before updating the model

        self.disentangle_distillation_loader = disentangle_distillation_loader
        self.eval_student = eval_student

    @classmethod
    def from_config(cls, cfg):
        student_backbone = build_backbone(cfg)
        student_sem_seg_head = build_sem_seg_head(cfg, student_backbone.output_shape())

        teacher_backbone = build_backbone(cfg)
        teacher_sem_seg_head = build_sem_seg_head(cfg, teacher_backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        kd_class_weight = cfg.MODEL.MASK_FORMER.KD_CLASS_WEIGHT  # cfg.MODEL.MASK_FORMER.CLASS_WEIGHT #cfg.MODEL.MASK_FORMER.KD_CLASS_WEIGHT
        kd_mask_weight = cfg.MODEL.MASK_FORMER.KD_MASK_WEIGHT  # cfg.MODEL.MASK_FORMER.MASK_WEIGHT #cfg.MODEL.MASK_FORMER.KD_MASK_WEIGHT
        kd_dice_weight = cfg.MODEL.MASK_FORMER.KD_DICE_WEIGHT  # cfg.MODEL.MASK_FORMER.DICE_WEIGHT #cfg.MODEL.MASK_FORMER.KD_DICE_WEIGHT

        num_predictions_distillation = cfg.MODEL.MASK_FORMER.NUM_PREDICTIONS_DISTILLATION

        num_predictions_inference = cfg.MODEL.MASK_FORMER.TEST.NUM_PREDICTIONS

        distillation_nms = cfg.MODEL.MASK_FORMER.DISTILLATION_NMS

        accum_iter = cfg.SOLVER.ACCUM_ITER

        # building criterion
        if any([class_weight > 0, mask_weight > 0, dice_weight > 0]):
            matcher = VideoHungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        else:
            matcher = VideoHungarianMatcher(
                cost_class=kd_class_weight,
                cost_mask=kd_mask_weight,
                cost_dice=kd_dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                       "kd_loss_ce": kd_class_weight,
                       "kd_loss_mask": kd_mask_weight, "kd_loss_dice": kd_dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.MODEL.MASK_FORMER.MASK_DROPLOSS:
            losses[1] = "masks_drop"
        if cfg.MODEL.MASK_FORMER.LABEL_DROPLOSS:
            losses[0] = "labels_drop"

        criterion = VideoSetCriterion(
            student_sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            loss_strategy=cfg.MODEL.MASK_FORMER.LOSS_STRATEGY,
            distillation_loss_strategy=cfg.MODEL.MASK_FORMER.DISTILLATION_LOSS_STRATEGY,
        )

        return {
            "student_backbone": student_backbone,
            "student_sem_seg_head": student_sem_seg_head,
            "teacher_backbone": teacher_backbone,
            "teacher_sem_seg_head": teacher_sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_predictions_distillation": num_predictions_distillation,
            "score_threshold_distillation": cfg.MODEL.MASK_FORMER.SCORE_THRESHOLD_DISTILLATION,
            "use_nms": cfg.MODEL.MASK_FORMER.TEST.USE_NMS,
            "nms_threshold": cfg.MODEL.MASK_FORMER.TEST.NMS_THRESH,
            "num_predictions_inference": num_predictions_inference,
            "distillation_nms": distillation_nms,
            "accum_iter": accum_iter,  # Number of iterations to accumulate gradients before updating the model
            "disentangle_distillation_loader": cfg.INPUT.DISENTANGLE_DISTILLATION_LOADER,
            "eval_student": cfg.MODEL.MASK_FORMER.TEST.EVAL_STUDENT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []

        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if self.training:
            if self.disentangle_distillation_loader:
                distill_images = []
                for video in batched_inputs:
                    for frame in video["distill_image"]:
                        distill_images.append(frame.to(self.device))
                distill_images = [(x - self.pixel_mean) / self.pixel_std for x in distill_images]
                distill_images = ImageList.from_tensors(distill_images, self.size_divisibility)

                distill_transformation_matrices = []
                for video in batched_inputs:
                    distill_transformation_matrices.append(video["distill_transforms_matrix"])

                distill_transforms_list = []
                for video in batched_inputs:
                    distill_transforms_list.append(video["distill_transforms"])

            student_outputs = self.student(images.tensor)
            if self.disentangle_distillation_loader:
                student_distillation_outputs = self.student(distill_images.tensor)

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    images.tensor)

                # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(student_outputs, targets)

            # KD loss
            distillation_targets = self.prepare_distillation_targets(teacher_outputs,
                                                                     images,
                                                                     None,
                                                                     nms=self.distillation_nms,
                                                                     score_threshold=self.score_threshold_distillation,
                                                                     visualization_images=images)

            distillation_losses = self.criterion(
                student_outputs if not self.disentangle_distillation_loader else student_distillation_outputs,
                distillation_targets, distillation=True)

            for k in list(distillation_losses.keys()):
                if k.startswith("loss_"):
                    new_k = k.replace("loss_", "kd_loss_")
                    distillation_losses[new_k] = distillation_losses.pop(k)

            losses.update(distillation_losses)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict and k in losses.keys():
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:

            with torch.no_grad():
                if self.eval_student:
                    teacher_outputs = self.student(images.tensor)
                else:
                    teacher_outputs = self.teacher(
                        images.tensor)

            mask_cls_results = teacher_outputs["pred_logits"]
            mask_pred_results = teacher_outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del teacher_outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]

        gt_instances = []
        for i, targets_per_video in enumerate(targets):
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])

                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def prepare_pseudo_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for i, targets_per_video in enumerate(targets):
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.shape[-2], targets_per_frame.shape[-1]

                if targets_per_frame.shape[-2] > gt_masks_per_video.shape[-2] or targets_per_frame.shape[-1] > \
                        gt_masks_per_video.shape[-1]:
                    print("WARNING: targets_per_frame shape dis too large, h,w, resizing")
                    h, w = gt_masks_per_video.shape[-2], gt_masks_per_video.shape[-1]
                    targets_per_frame = F.interpolate(targets_per_frame.unsqueeze(1).float(), size=(h, w),
                                                      mode="bilinear", align_corners=False).squeeze(1) > 0.5

                    print("targets_per_frame.gt_masks.tensor.shape after resizing", targets_per_frame.shape)

                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame

            print("targets_per_video", targets_per_video)

            gt_instances.append({"masks": gt_masks_per_video, "labels": targets_per_video["labels"],
                                 "scores": targets_per_video["scores"]})

        return gt_instances

    def prepare_distillation_targets(self, teacher_outputs, images, transformation_matrices, nms=False,
                                     score_threshold=0.75, visualization_images=None):
        """
        Prepare distillation targets from the teacher model outputs.
        Args:
            teacher_outputs: Outputs from the teacher model.
            images: Input images for which the distillation targets are prepared.
            transformation_matrices: List of transformation matrices for each video frame.
            nms: Whether to apply Non-Maximum Suppression (NMS) on the masks.
        Returns:
            List of dictionaries containing distillation targets for each video frame.
        """
        mask_cls_results = teacher_outputs["pred_logits"]
        mask_pred_results = teacher_outputs["pred_masks"]

        h_pad, w_pad = images.tensor.shape[-2:]
        distillation_targets = []

        for i in range(mask_pred_results.shape[0]):
            video_mask_cls_result = mask_cls_results[i]
            if len(video_mask_cls_result) > 0:
                scores = F.softmax(video_mask_cls_result, dim=-1)[:, :-1]
                labels = torch.arange(self.teacher[1].num_classes, device=self.device).unsqueeze(0).repeat(
                    self.num_queries, 1).flatten(0, 1)
                # keep top-10 predictions
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_predictions_distillation,
                                                                           sorted=False)
                labels_per_image = labels[topk_indices]
                topk_indices = topk_indices // self.teacher[1].num_classes

                video_mask_pred_results = mask_pred_results[i]

                if topk_indices.device != mask_pred_results.device:
                    video_mask_pred_results = video_mask_pred_results.to(topk_indices.device)

                video_mask_pred_results = video_mask_pred_results[topk_indices]

                # Apply score threshold
                score_mask = scores_per_image >= score_threshold
                scores_per_image = scores_per_image[score_mask]
                labels_per_image = labels_per_image[score_mask]

                video_mask_pred_results = video_mask_pred_results[score_mask]

                # video_mask_pred_results = video_mask_pred_results[:, :, : img_size[0], : img_size[1]]
                video_mask_pred_results = video_mask_pred_results[:, :, : h_pad, : w_pad]
                video_mask_pred_results = F.interpolate(
                    video_mask_pred_results, size=(h_pad, w_pad), mode="bilinear", align_corners=False
                )

                video_mask_pred_masks = video_mask_pred_results > 0.

                if transformation_matrices is not None:
                    per_frame_transformation_matrices = transformation_matrices[i]

                    # mask_shape = [self.num_predictions_distillation, self.num_frames, h_pad, w_pad]
                    mask_shape = [video_mask_pred_masks.shape[0], self.num_frames, h_pad, w_pad]
                    gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                    ###transformed_video_pred_masks = apply_transformation_frame_by_frame(video_mask_pred_masks, per_frame_transformation_matrices, gt_masks_per_video)
                    gt_masks_per_video = apply_transformation_frame_by_frame(video_mask_pred_masks,
                                                                             per_frame_transformation_matrices,
                                                                             gt_masks_per_video)
                else:
                    gt_masks_per_video = video_mask_pred_masks

                if nms:
                    # Apply NMS to the mask predictions
                    keep = []

                    indices = list(range(len(scores_per_image)))

                    while len(indices) > 0:
                        current_idx = indices.pop(0)
                        keep.append(current_idx)

                        remaining_indices = []
                        current_mask = gt_masks_per_video[current_idx]
                        current_label = labels_per_image[current_idx]

                        for other_idx in indices:
                            other_mask = gt_masks_per_video[other_idx]
                            other_label = labels_per_image[other_idx]

                            # Wenden Sie NMS nur für Masken derselben Klasse an
                            if current_label != other_label:
                                remaining_indices.append(other_idx)
                                continue

                            intersection = torch.sum(current_mask & other_mask).float()
                            union = torch.sum(current_mask | other_mask).float()

                            iou = intersection / union if union > 0 else 0.0

                            if iou <= self.nms_threshold:
                                remaining_indices.append(other_idx)

                        indices = remaining_indices

                    # Filtern Sie die Ausgaben basierend auf den NMS-Ergebnissen
                    gt_masks_per_video = gt_masks_per_video[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

            distillation_targets.append({
                "masks": gt_masks_per_video,
                "masks_logits": mask_pred_results[i][topk_indices],
                "labels": labels_per_image,
            })

        return distillation_targets

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.teacher[1].num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                                                                       1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_predictions_inference, sorted=True)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.teacher[1].num_classes

            if topk_indices.device != pred_masks.device:
                pred_masks = pred_masks.to(topk_indices.device)

            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            if self.use_nms:
                keep = []

                indices = list(range(len(scores_per_image)))

                while len(indices) > 0:
                    current_idx = indices.pop(0)
                    keep.append(current_idx)

                    remaining_indices = []
                    current_mask = masks[current_idx]
                    current_label = labels_per_image[current_idx]

                    for other_idx in indices:
                        other_mask = masks[other_idx]
                        other_label = labels_per_image[other_idx]

                        # Wenden Sie NMS nur für Masken derselben Klasse an
                        if current_label != other_label:
                            remaining_indices.append(other_idx)
                            continue

                        intersection = torch.sum(current_mask & other_mask).float()
                        union = torch.sum(current_mask | other_mask).float()

                        iou = intersection / union if union > 0 else 0.0

                        if iou <= self.nms_threshold:
                            remaining_indices.append(other_idx)

                    indices = remaining_indices

                # Filtern Sie die Ausgaben basierend auf den NMS-Ergebnissen
                final_masks = masks[keep]
                final_scores = scores_per_image[keep]
                final_labels = labels_per_image[keep]

                out_scores = final_scores.tolist()
                out_labels = final_labels.tolist()
                out_masks = [m for m in final_masks.cpu()]

            else:
                out_scores = scores_per_image.tolist()
                out_labels = labels_per_image.tolist()
                out_masks = [m for m in masks.cpu()]

        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
