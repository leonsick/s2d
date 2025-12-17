# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py

import torch
from math import inf

from detectron2.utils.events import get_event_storage
from torch.nn.parallel import DataParallel, DistributedDataParallel
import math

import numpy as np
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
import copy
import random
import torch.nn.functional as F
from detectron2.structures.instances import Instances
from detectron2.structures import BitMasks

from detectron2.engine import SimpleTrainer
import detectron2.utils.comm as comm

from ..data_video.dataset_mapper import YTVISDatasetMapper
from ..data_video.build import get_detection_dataset_dicts, build_detection_train_loader


# --- Helper to densify sparse YTVIS masks by copying preceding frame masks ---
def propagate_sparse_masks(instances_per_frame, max_shift=2):
    # Fill sparse video masks by copying the most recent preceding frame's mask
    # for the same instance id (instances.gt_ids) into frames where it's missing.
    # Only a tiny translation jitter (±max_shift px) is applied.

    if not instances_per_frame:
        return instances_per_frame

    out = [copy.deepcopy(inst) for inst in instances_per_frame]
    T = len(out)

    def _ensure_core_fields(inst):
        # Make sure gt_masks / boxes / classes exist, even if empty.
        H, W = inst._image_size if hasattr(inst, "_image_size") else inst.image_size
        if not inst.has("gt_masks"):
            inst.set("gt_masks", BitMasks(torch.zeros((0, H, W), dtype=torch.bool)))
        if not inst.has("gt_boxes"):
            inst.set("gt_boxes", inst.gt_masks.get_bounding_boxes())
        if not inst.has("gt_classes"):
            inst.set("gt_classes", torch.zeros((len(inst),), dtype=torch.long))

    def _tensor_masks(inst):
        # Always returns (N,H,W) bool tensor
        m = inst.gt_masks
        if isinstance(m, BitMasks):
            return m.tensor.to(torch.bool)
        return m.to(torch.bool)

    def _translate(mask_hw, dx, dy):
        # mask_hw: (H,W)
        H, W = mask_hw.shape[-2], mask_hw.shape[-1]
        outm = torch.zeros_like(mask_hw)
        xs = slice(max(0, dx), min(W, W + dx))
        xt = slice(max(0, -dx), min(W, W - dx))
        ys = slice(max(0, dy), min(H, H + dy))
        yt = slice(max(0, -dy), min(H, H - dy))
        if (xt.stop - xt.start) > 0 and (yt.stop - yt.start) > 0:
            outm[yt, xt] = mask_hw[ys, xs]
        return outm

    # Sanity + early exit if ids not present
    for inst in out:
        _ensure_core_fields(inst)
        if not inst.has("gt_ids"):
            return out  # nothing we can do if there are no ids

    # last seen mask & class for each id (from preceding frames only)
    last_seen = {}  # tid -> {"mask": (H,W) bool tensor, "cls": int}

    for t in range(T):
        inst = out[t]
        H, W = inst._image_size if hasattr(inst, "_image_size") else inst.image_size
        masks_t = _tensor_masks(inst)      # (N,H,W) or (0,H,W)
        ids_t   = inst.get("gt_ids")       # (N,)
        cls_t   = inst.get("gt_classes")   # (N,)

        # 1) update cache with whatever is present in this frame
        if len(inst):
            for i in range(len(inst)):
                tid = int(ids_t[i].item())
                if masks_t.numel() > 0:
                    last_seen[tid] = {
                        "mask": masks_t[i],
                        "cls": int(cls_t[i].item()) if i < len(cls_t) else None,
                    }

        # 2) synthesize any ids we've already seen but are missing *in this frame*
        present_ids = set(int(x) for x in ids_t.tolist()) if len(inst) else set()
        to_fill = [tid for tid, pack in last_seen.items() if tid not in present_ids and pack["mask"] is not None]
        if not to_fill:
            continue

        # Gather existing fields to rebuild Instances cleanly
        extra_fields = {k: v for k, v in inst.get_fields().items() if k not in ["gt_masks", "gt_boxes", "gt_classes", "gt_ids"]}

        mask_list = []
        if masks_t.numel() > 0:
            mask_list.append(masks_t)

        new_classes = []
        new_ids = []

        for tid in to_fill:
            prev = last_seen[tid]
            dx = random.randint(-max_shift, max_shift) if max_shift > 0 else 0
            dy = random.randint(-max_shift, max_shift) if max_shift > 0 else 0
            pasted = _translate(prev["mask"], dx, dy)[None, ...]  # (1,H,W)
            mask_list.append(pasted)
            new_ids.append(tid)
            if prev["cls"] is not None:
                new_classes.append(prev["cls"])

        all_masks = torch.cat(mask_list, dim=0) if mask_list else torch.zeros((0, H, W), dtype=torch.bool)

        # Rebuild instances for this frame
        new_inst = Instances(inst._image_size if hasattr(inst, "_image_size") else inst.image_size)
        new_inst.set("gt_masks", BitMasks(all_masks))
        new_inst.set("gt_boxes", BitMasks(all_masks).get_bounding_boxes())

        # classes
        if len(inst):
            if new_classes:
                new_cls = torch.tensor(new_classes, dtype=cls_t.dtype, device=cls_t.device)
                new_inst.set("gt_classes", torch.cat([cls_t, new_cls], dim=0))
            else:
                new_inst.set("gt_classes", cls_t)
        else:
            # frame had no instances; create just the new ones
            new_inst.set("gt_classes", torch.tensor(new_classes, dtype=torch.long) if new_classes else torch.zeros((0,), dtype=torch.long))

        # ids
        if len(inst):
            if new_ids:
                new_ids_t = torch.tensor(new_ids, dtype=ids_t.dtype, device=ids_t.device)
                new_inst.set("gt_ids", torch.cat([ids_t, new_ids_t], dim=0))
            else:
                new_inst.set("gt_ids", ids_t)
        else:
            new_inst.set("gt_ids", torch.tensor(new_ids, dtype=torch.long) if new_ids else torch.zeros((0,), dtype=torch.long))

        # put back any extra fields unchanged (e.g., iscrowd, etc.)
        for k, v in extra_fields.items():
            new_inst.set(k, v)

        out[t] = new_inst

    return out
# --- end helper ---

def linear_weight_update(weight, step, start_step, end_step, min_weight, kd):
    total_steps = end_step - start_step
    step = step - start_step
    q = step / total_steps  # Note: Changed to float division for continuous range
    weight_range = weight - min_weight

    if q < 0:
        q = 0

    # Calculate the fraction of the weight range to add/subtract
    # If normal loss, the weight should go from `weight` down to `min_weight`.
    # The factor for this is (1 - q), which goes from 1 to 0.
    # If kd is true, the weight should go from `min_weight` up to `weight`.
    # The factor for this is q, which goes from 0 to 1.
    weight_change_factor = (1 - q) if not kd else q

    weight_change = weight_range * weight_change_factor

    new_weight = min_weight + weight_change # The change is always added to min_weight to ensure it stays in the range [min_weight, weight]

    return new_weight

def cosine_weight_update(weight, step, start_step, end_step, min_weight, kd):
    """
    Updates a weight parameter using a cosine schedule, which provides a smoother
    transition than a linear schedule.

    Args:
        weight (float): The initial maximum weight (or final maximum if kd is False).
        step (int): The current training step.
        total_steps (int): The total number of steps in the schedule.
        min_weight (float): The minimum weight (or starting minimum if kd is False).
        kd (bool): If True, the weight decays from 'weight' to 'min_weight' (cosine decay).
                   If False, the weight increases from 'min_weight' to 'weight' (cosine ramp-up).

    Returns:
        float: The new weight.
    """
    total_steps = end_step - start_step
    step = step - start_step
    # Normalize the step to a [0, 1] range
    q = step / total_steps

    if q < 0:
        q = 0

    # The cosine interpolation factor goes from 1 to 0 over the range [0, total_steps].
    # formula: 0.5 * (1 + cos(pi * q))
    # At q=0, factor = 0.5 * (1 + cos(0)) = 0.5 * 2 = 1
    # At q=1, factor = 0.5 * (1 + cos(pi)) = 0.5 * (1 - 1) = 0
    cosine_decay_factor = 0.5 * (1 + math.cos(math.pi * q))

    weight_range = weight - min_weight

    if not kd:
        # If normal loss
        # Cosine decay: Factor goes from 1 to 0.
        # At step=0: weight_range * 1 + min_weight = weight
        # At step=total_steps: weight_range * 0 + min_weight = min_weight
        new_weight = weight_range * cosine_decay_factor + min_weight
    else:
        # If KD
        # Cosine ramp-up: We want the factor to go from 0 to 1.
        # Use (1 - cosine_decay_factor) for the ramp-up.
        # At step=0: weight_range * 0 + min_weight = min_weight
        # At step=total_steps: weight_range * 1 + min_weight = weight
        cosine_ramp_up_factor = 1.0 - cosine_decay_factor
        new_weight = weight_range * cosine_ramp_up_factor + min_weight

    return new_weight

def update_loss_weights(update_fn, original_weight_dict, step, start_step, end_step, minimum_weight_dict):
    """
    weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "kd_loss_ce": kd_class_weight,
                       "kd_loss_mask": kd_mask_weight, "kd_loss_dice": kd_dice_weight}
    """
    new_weight_dict = copy.deepcopy(original_weight_dict)
    for k in new_weight_dict.keys():
        new_weight_dict[k] = update_fn(original_weight_dict[k], step, start_step, end_step, minimum_weight_dict[k], kd=("kd" in k))

    return new_weight_dict


__all__ = ["CustomSimpleTrainer", "CustomAMPTrainer"]

class CustomSimpleTrainer(SimpleTrainer):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, cfg=None, use_copy_paste=False,
                copy_paste_rate=-1, copy_paste_random_num=None, copy_paste_min_ratio=-1,
                copy_paste_max_ratio=-1, visualize_copy_paste=False):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__(model, data_loader, optimizer)

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        self.cfg = cfg
        # model.train() #

        # self.model = model
        # self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        # self._data_loader_iter_obj = None
        # self.optimizer = optimizer

        self.use_copy_paste = use_copy_paste if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE
        self.cfg_COPY_PASTE_RATE = copy_paste_rate if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_RATE
        self.cfg_COPY_PASTE_RANDOM_NUM = copy_paste_random_num if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_RANDOM_NUM
        self.cfg_COPY_PASTE_MIN_RATIO = copy_paste_min_ratio if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_MIN_RATIO
        self.cfg_COPY_PASTE_MAX_RATIO = copy_paste_max_ratio if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_MAX_RATIO
        self.cfg_VISUALIZE_COPY_PASTE = visualize_copy_paste if self.cfg is None else self.cfg.DATALOADER.VISUALIZE_COPY_PASTE
        self.cfg_COPY_PASTE_DENSIFY_SPARSE = False if self.cfg is None else self.cfg.DATALOADER.COPY_PASTE_DENSIFY_SPARSE
        print("copy_paste hyper-params:", self.use_copy_paste, self.cfg_COPY_PASTE_RATE, self.cfg_COPY_PASTE_RANDOM_NUM, "densify_sparse:", self.cfg_COPY_PASTE_DENSIFY_SPARSE)

        self.training_max_iter = self.cfg.SOLVER.MAX_ITER if self.cfg is not None else 90000
        self.training_iter_steps = self.cfg.SOLVER.STEPS if self.cfg is not None else (60000, 80000)

        if self.cfg.MODEL.META_ARCHITECTURE == "KDVideoMaskFormer":
            self.m = self.cfg.MODEL.MASK_FORMER.EMA_MOMENTUM if self.cfg is not None else 0.999

            self.m_schedule = self.cfg.MODEL.MASK_FORMER.EMA_MOMENTUM_SCHEDULE if self.cfg is not None else False
            self.m_start = self.cfg.MODEL.MASK_FORMER.EMA_MOMENTUM if self.cfg is not None else 0.999
            self.m_end = self.cfg.MODEL.MASK_FORMER.EMA_MOMENTUM_END if self.cfg is not None else 0.999
            self.m_end_iter = self.cfg.MODEL.MASK_FORMER.EMA_MOMENTUM_UNTIL_STEP if self.cfg is not None else 10000


            self.loss_weight_decay_step = int(self.cfg.MODEL.MASK_FORMER.LOSS_WEIGHT_DECAY_STEP) if self.cfg is not None else 0
            if self.loss_weight_decay_step > 0:
                print("INFO: Decay loss weight at step {}.".format(self.loss_weight_decay_step))
                time.sleep(2)


            self.kd_weight_scheduler = self.cfg.MODEL.MASK_FORMER.KD_WEIGHT_SCHEDULER
            self.kd_schedule_min_weight = self.cfg.MODEL.MASK_FORMER.KD_MIN_WEIGHT
            self.supervised_schedule_min_weight = self.cfg.MODEL.MASK_FORMER.SUPERVISED_MIN_WEIGHT

            self.schedule_start_step = int(self.cfg.MODEL.MASK_FORMER.KD_WEIGHT_DECAY_START) if self.cfg is not None else 0
            self.schedule_end_step = int(self.cfg.MODEL.MASK_FORMER.KD_WEIGHT_DECAY_END) if self.cfg is not None else -1

            if self.schedule_end_step == -1:
                self.schedule_end_step = self.training_max_iter

            self.decay_only_supervised_loss = self.cfg.MODEL.MASK_FORMER.DECAY_ONLY_SUPERVISED_LOSS
            self.decay_only_kd_loss = self.cfg.MODEL.MASK_FORMER.DECAY_ONLY_KD_LOSS

            print(f"Using KD weight scheduler {self.kd_weight_scheduler} with KD Min Weight {self.kd_schedule_min_weight} and Supervised Min Weight {self.supervised_schedule_min_weight}.")
            print(f"Weight schedule starts at step {self.schedule_start_step} and ends at step {self.schedule_end_step}.")
            print(f"Training max iteration is set to {self.training_max_iter}.")
            print(f"Decaying only the supervised loss: {self.cfg.MODEL.MASK_FORMER.DECAY_ONLY_SUPERVISED_LOSS}")

            self.original_mask_weight = self.cfg.MODEL.MASK_FORMER.MASK_WEIGHT
            self.original_dice_weight = self.cfg.MODEL.MASK_FORMER.MASK_WEIGHT
            self.original_class_weight = self.cfg.MODEL.MASK_FORMER.CLASS_WEIGHT

            self.original_kd_mask_weight = self.cfg.MODEL.MASK_FORMER.KD_MASK_WEIGHT
            self.original_kd_dice_weight = self.cfg.MODEL.MASK_FORMER.KD_MASK_WEIGHT
            self.original_kd_class_weight = self.cfg.MODEL.MASK_FORMER.KD_CLASS_WEIGHT

            """self.original_weight_dict = {"loss_ce": self.original_class_weight, "loss_mask": self.original_mask_weight,
                                         "loss_dice": self.original_dice_weight,
                                         "kd_loss_ce": self.original_kd_class_weight,
                                         "kd_loss_mask": self.original_kd_mask_weight,
                                         "kd_loss_dice": self.original_kd_dice_weight}

            self.minimum_weight_dict = {"loss_ce": self.original_class_weight*self.schedule_min_weight, "loss_mask": self.original_mask_weight*self.schedule_min_weight,
                                         "loss_dice": self.original_dice_weight*self.schedule_min_weight,
                                         "kd_loss_ce": self.original_kd_class_weight*self.schedule_min_weight,
                                         "kd_loss_mask": self.original_kd_mask_weight*self.schedule_min_weight,
                                         "kd_loss_dice": self.original_kd_dice_weight*self.schedule_min_weight}"""

            self.original_weight_dict = copy.deepcopy(self.model.criterion.weight_dict)

            self.minimum_weight_dict = {
                k: v*self.supervised_schedule_min_weight if ("kd" not in k) else v*self.kd_schedule_min_weight
                for k, v in self.original_weight_dict.items()}




    def IoU(self, mask1, mask2): # only work when the batch size is 1
        mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
        intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
        union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
        return (intersection.to(torch.float) / union).mean().view(1, -1)

    def IoY(self, mask1, mask2): # only work when the batch size is 1
        mask1, mask2 = mask1.squeeze(), mask2.squeeze()
        mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
        intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
        union = torch.sum(mask2, dim=[-1, -2]).squeeze()
        return (intersection.to(torch.float) / union).mean().view(1, -1)

    def copy_and_paste(self, sources, targets):
        def mask_iou_matrix(x, y, mode='iou'):
            x = x.reshape(x.shape[0], -1).float()
            y = y.reshape(y.shape[0], -1).float()
            inter_matrix = x @ y.transpose(1, 0) # n1xn2
            sum_x = x.sum(1)[:, None].expand(x.shape[0], y.shape[0])
            sum_y = y.sum(1)[None, :].expand(x.shape[0], y.shape[0])
            if mode == 'ioy':
                iou_matrix = inter_matrix / (sum_y) # [1, 1]
            else:
                iou_matrix = inter_matrix / (sum_x + sum_y - inter_matrix) # [1, 1]
            return iou_matrix

        def visualize_data(data, save_path = './sample.jpg'):
            from detectron2.data import detection_utils as utils
            from detectron2.data import DatasetCatalog, MetadataCatalog
            from detectron2.utils.visualizer import Visualizer
            data["instances"] = data["instances"].to(device='cpu')
            img = data["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, 'RGB')
            metadata = MetadataCatalog.get('ytvis_2019_train_cls_agnostic')
            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = data["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes"), # ("gt_boxes", None),
                    masks=target_fields.get("gt_masks"), # ("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
            )
            print("Saving to {} ...".format(save_path))
            vis.save(save_path)

        new_targets = []
        target_data_copy = []
        for source_data, target_data in zip(sources, targets):
            # data dict is created by mask2former_video/data_video/dataset_mapper.py
            # data['instances']: list of instances [[frame1 instances],[frame2 instances]...[frameN instances]]
            source_instances = source_data["instances"]
            source_image_list = source_data["image"]
            target_instances_list = target_data["instances"]
            target_image_list = target_data["image"]
            target_data_copy.append(copy.deepcopy(target_data))

            num_source_instances = len(source_instances[0])
            copy_paste_rate = random.random()
            if self.cfg_COPY_PASTE_RATE >= copy_paste_rate and num_source_instances > 0:
                if self.cfg_COPY_PASTE_RANDOM_NUM:
                    num_copy = 1 if num_source_instances == 1 else np.random.randint(1, max(1, num_source_instances))
                else:
                    num_copy = num_source_instances
            else:
                num_copy = 0

            if num_copy > 0 and self.cfg_COPY_PASTE_DENSIFY_SPARSE:
                # Fill sparse masks forward per id to keep T consistent
                try:
                    target_data["instances"] = propagate_sparse_masks(target_data["instances"], max_shift=2)
                except Exception as _e:
                    pass
                new_targets.append(target_data)
                continue
            else:
                choice = np.random.choice(num_source_instances, num_copy, replace=False)
                # randomly choose instances from the first frame and copy all these selected instances
                frame_id = np.random.randint(1, max(1, len(source_instances))) - 1
                copied_instances = source_instances[frame_id][choice].to(device=target_instances_list[frame_id].gt_boxes.device)

                # paste these instances to ALL frames in the same video
                target_instances_list_new = []
                target_image_list_new = []
                for f in range(len(target_instances_list)):
                    # Use DEEPCOPY, otherwise, the objects will be in-place edited
                    target_instances = copy.deepcopy(target_instances_list[f])
                    target_image = copy.deepcopy(target_image_list[f])
                    copied_masks = copy.deepcopy(copied_instances.gt_masks)
                    copied_boxes = copy.deepcopy(copied_instances.gt_boxes)
                    _, labeled_h, labeled_w = source_image_list[frame_id].shape
                    _, unlabeled_h, unlabeled_w = target_image.shape

                    # rescale the labeled image to align with unlabeled one.
                    if isinstance(copied_masks, torch.Tensor):
                        masks_new = copied_masks[None, ...].float()
                    else:
                        masks_new = copied_masks.tensor[None, ...].float()
                    # resize the masks with a random ratio from 0.5 to 1.0
                    resize_ratio = random.uniform(self.cfg_COPY_PASTE_MIN_RATIO, self.cfg_COPY_PASTE_MAX_RATIO)
                    w_new = int(resize_ratio * unlabeled_w)
                    h_new = int(resize_ratio * unlabeled_h)

                    # randomly shift the masks，so that the masks are not always in the center of the image
                    w_shift = random.randint(0, max(0, unlabeled_w - w_new))
                    h_shift = random.randint(0, max(0, unlabeled_h - h_new))

                    source_image_new = F.interpolate(source_image_list[frame_id][None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).byte().squeeze(0)

                    if isinstance(copied_masks, torch.Tensor):
                        if copied_masks.shape[0] == 0:
                            # Append the original image and instances if no masks are copied
                            target_image_list_new.append(target_image)
                            target_instances_list_new.append(target_instances)
                            continue
                    else:
                        if copied_masks.tensor.shape[0] == 0:
                            # Append the original image and instances if no masks are copied
                            target_image_list_new.append(target_image)
                            target_instances_list_new.append(target_instances)
                            continue

                    if isinstance(copied_masks, torch.Tensor):
                        masks_new = F.interpolate(copied_masks[None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).bool().squeeze(0)
                    else:
                        masks_new = F.interpolate(copied_masks.tensor[None, ...].float(), size=(h_new, w_new), mode="bilinear", align_corners=False).bool().squeeze(0)
                    copied_boxes.scale(1. * unlabeled_w / labeled_w * resize_ratio, 1. * unlabeled_h / labeled_h * resize_ratio)

                    if isinstance(target_instances.gt_masks, torch.Tensor):
                        _, mask_w, mask_h = target_instances.gt_masks.size()
                    else:
                        _, mask_w, mask_h = target_instances.gt_masks.tensor.size()

                    masks_new_all = torch.zeros(num_copy, mask_w, mask_h)
                    image_new_all = torch.zeros_like(target_image)

                    image_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += source_image_new
                    masks_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += masks_new

                    source_image = image_new_all.byte() #.squeeze(0)
                    if isinstance(copied_masks, torch.Tensor):
                        copied_masks = masks_new_all.bool() #.squeeze(0)
                    else:
                        copied_masks.tensor = masks_new_all.bool() #.squeeze(0)
                    copied_boxes.tensor[:, 0] += h_shift
                    copied_boxes.tensor[:, 2] += h_shift
                    copied_boxes.tensor[:, 1] += w_shift
                    copied_boxes.tensor[:, 3] += w_shift

                    copied_instances.gt_masks = copied_masks
                    copied_instances.gt_boxes = copied_boxes
                    copied_instances._image_size = (unlabeled_h, unlabeled_w)
                    if len(target_instances) == 0:
                        if isinstance(copied_instances.gt_masks, torch.Tensor):
                            alpha = copied_instances.gt_masks.sum(0) > 0
                        else:
                            alpha = copied_instances.gt_masks.tensor.sum(0) > 0
                        # merge image
                        alpha = alpha.cpu()
                        composited_image = (alpha * source_image) + (~alpha * target_image)

                        target_image_list_new.append(composited_image)
                        target_instances_list_new.append(copied_instances)
                    else:
                        # remove the copied object if iou greater than 0.5
                        if isinstance(copied_masks, torch.Tensor):
                            iou_matrix = mask_iou_matrix(copied_masks, target_instances.gt_masks, mode='ioy') # nxN
                        else:
                            iou_matrix = mask_iou_matrix(copied_masks.tensor, target_instances.gt_masks.tensor, mode='ioy') # nxN

                        # check if the iou is greater than 0.5. for each video, all frames should have
                        # the same amount of instances and gt masks (can be None).
                        if f == 0:
                            keep = iou_matrix.max(1)[0] < 0.5
                            sum_keep = keep.sum()
                        else:
                            keep = iou_matrix.max(1)[0] < 2.0

                        if sum_keep < keep.size()[0]:
                            target_image_list_new.append(target_image)
                            target_instances_list_new.append(target_instances)
                            continue

                        copied_instances = copied_instances[keep]
                        # update existing instances in unlabeled image
                        if isinstance(copied_instances.gt_masks, torch.Tensor):
                            alpha = copied_instances.gt_masks.sum(0) > 0
                            target_instances.gt_masks = ~alpha * target_instances.gt_masks
                            areas_unlabeled = target_instances.gt_masks.sum((1,2))
                        else:
                            alpha = copied_instances.gt_masks.tensor.sum(0) > 0
                            target_instances.gt_masks.tensor = ~alpha * target_instances.gt_masks.tensor
                            areas_unlabeled = target_instances.gt_masks.tensor.sum((1,2))
                        # merge image
                        alpha = alpha.cpu()
                        composited_image = (alpha * source_image) + (~alpha * target_image)
                        # merge instances
                        merged_instances = Instances.cat([target_instances[areas_unlabeled > 0], copied_instances])
                        # update boxes
                        if isinstance(merged_instances.gt_masks, torch.Tensor):
                            merged_instances.gt_boxes = BitMasks(merged_instances.gt_masks).get_bounding_boxes()
                        else:
                            merged_instances.gt_boxes = merged_instances.gt_masks.get_bounding_boxes()

                        target_image_list_new.append(composited_image)
                        target_instances_list_new.append(merged_instances)

                    if self.cfg_VISUALIZE_COPY_PASTE:
                        visualize_data(target_data, save_path = 'sample_{}.jpg'.format(np.random.randint(5)))

                target_data["image"] = target_image_list_new
                # Fill sparse masks forward per id to keep T consistent
                try:
                    target_data["instances"] = propagate_sparse_masks(target_instances_list_new, max_shift=2)
                except Exception as _e:
                    target_data["instances"] = target_instances_list_new
                new_targets.append(target_data)

        new_targets_outputs = []
        for t, new_target in enumerate(new_targets):
            n_insts = [len(new_target["instances"][i]) for i in range(len(new_target["instances"]))]
            try:
                assert len(set(n_insts)) == 1, "The number of instances should be the same for all frames in the same video."
                new_targets_outputs.append(new_target)
            except:
                new_targets_outputs.append(target_data_copy[t])
        return new_targets_outputs

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        if self.use_copy_paste:
            data = self.copy_and_paste(copy.deepcopy(data[::-1]), data)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if not torch.isnan(losses):
            self.optimizer.zero_grad()
            losses.backward()
        else:
            print('Nan loss. Skipped.')

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()


class CustomAMPTrainer(CustomSimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, cfg=None, grad_scaler=None, use_copy_paste=False, 
                copy_paste_rate=-1, copy_paste_random_num=None, copy_paste_min_ratio=-1, 
                copy_paste_max_ratio=-1, visualize_copy_paste=False):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported
        print("INFO: use AMPTrainer.")

        super().__init__(model, data_loader, optimizer, cfg=cfg, use_copy_paste=use_copy_paste, \
            copy_paste_rate=copy_paste_rate, copy_paste_random_num=copy_paste_random_num, \
            copy_paste_min_ratio=copy_paste_min_ratio, copy_paste_max_ratio=copy_paste_max_ratio, \
            visualize_copy_paste=visualize_copy_paste)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        if self.model.accum_iter > 1:
            self.grad_accum_scaler = NativeScalerWithGradNormCount()

    def custom_reset_data_loader(self, data_loader_builder, cfg):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder(cfg)
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)

        if self.cfg.MODEL.META_ARCHITECTURE == "KDVideoMaskFormer":
            if self.iter == self.loss_weight_decay_step and self.loss_weight_decay_step > 0:
                print("INFO: Decay loss weights at step {}.".format(self.iter))
                for k in self.model.criterion.weight_dict.keys():
                    if not k.__contains__("kd_"):
                        # This used to be fixed 0.1 before, now its a cfg parameter
                        self.model.criterion.weight_dict[k] *= self.supervised_schedule_min_weight
                        print("INFO: Decay loss weight {} to {}".format(k, self.model.criterion.weight_dict[k]))


        if self.use_copy_paste:
            data = self.copy_and_paste(copy.deepcopy(data[::-1]), data)
        data_time = time.perf_counter() - start


        if self.model.accum_iter <= 1:
            # Without gradient accumulation
            with autocast():
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            if not torch.isnan(losses):
                self.optimizer.zero_grad()
                self.grad_scaler.scale(losses).backward()
            else:
                print('Nan loss.')

            self._write_metrics(loss_dict, data_time)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        else:
            # Now with gradient accumulation
            with autocast():
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            if not torch.isnan(losses):
                losses /= self.model.accum_iter
                self.grad_accum_scaler(losses, self.optimizer,
                                       parameters=self.model.student.parameters() if hasattr(self.model, "student") else self.model.parameters(),
                                       update_grad=(self.iter + 1) % self.model.accum_iter == 0,)

                if (self.iter + 1) % self.model.accum_iter == 0:
                    self.optimizer.zero_grad()
            else:
                print('Nan loss.')

            self._write_metrics(loss_dict, data_time)


        # If we train with knowledge distillation, update the teacher model with EMA
        # If the teacher is a ddp model, use model.teacher.module as the target to be updated.
        if hasattr(self.model, "teacher") and self.model.teacher is not None:
            if isinstance(self.model.teacher, DistributedDataParallel):
                teacher_no_ddp = self.model.teacher.module
            else:
                teacher_no_ddp = self.model.teacher

            # Update the teacher model with EMA
            with torch.no_grad():
                if (self.iter + 1) % self.model.accum_iter == 0:
                    for param, teacher_param in zip(self.model.student.parameters(), teacher_no_ddp.parameters()):
                        teacher_param.data.mul_(self.m).add_((1 - self.m) * param.detach().data)

            # Update the momentum value if using momentum schedule
            if self.m_schedule:
                previous_m = self.m
                self.m = self.m_end - (self.m_end - self.m_start) * (math.cos(math.pi * (self.iter*self.model.accum_iter) / (self.m_end_iter*self.model.accum_iter)) + 1) / 2
                print("Update EMA momentum from {:.6f} to {:.6f} at step {}".format(previous_m, self.m, self.iter)) if self.iter % 1000 == 0 else None


            # TODO: Implement KD Weight Scheduler
            #if self.schedule_start_step <= self.iter <= self.schedule_end_step and self.kd_weight_scheduler in ["linear", "cosine"]:

            if self.kd_weight_scheduler in ["linear", "cosine"]:
                # Check if we decay only the supervised loss
                if self.decay_only_supervised_loss:
                    # Only decay the supervised loss weights
                    supervised_weight_dict = {k: v for k, v in self.original_weight_dict.items() if not k.__contains__("kd_")}
                    self.model.criterion.weight_dict.update(
                        update_loss_weights(
                            linear_weight_update if self.kd_weight_scheduler == "linear" else cosine_weight_update,
                            supervised_weight_dict,
                            self.iter, self.schedule_start_step, self.schedule_end_step,
                            {k: v for k, v in self.minimum_weight_dict.items() if not k.__contains__("kd_")}
                        )
                    )
                elif self.decay_only_kd_loss:
                    # Only decay the KD loss weights
                    kd_weight_dict = {k: v for k, v in self.original_weight_dict.items() if k.__contains__("kd_")}
                    self.model.criterion.weight_dict.update(
                        update_loss_weights(
                            linear_weight_update if self.kd_weight_scheduler == "linear" else cosine_weight_update,
                            kd_weight_dict,
                            self.iter, self.schedule_start_step, self.schedule_end_step,
                            {k: v for k, v in self.minimum_weight_dict.items() if k.__contains__("kd_")}
                        )
                    )
                else:
                    # Update the loss weights
                    if self.kd_weight_scheduler == "linear":
                        self.model.criterion.weight_dict = update_loss_weights(
                            linear_weight_update, self.original_weight_dict,
                            self.iter, self.schedule_start_step, self.schedule_end_step, self.minimum_weight_dict
                        )
                    elif self.kd_weight_scheduler == "cosine":
                        self.model.criterion.weight_dict = update_loss_weights(
                            cosine_weight_update, self.original_weight_dict,
                            self.iter, self.schedule_start_step, self.schedule_end_step, self.minimum_weight_dict
                        )
                    else:
                        pass

            # Log weight dict by adding weight/ to every key in separate dict
            weight_log_dict = {f"weight/{k}": v for k, v in self.model.criterion.weight_dict.items()}

            # Log the weights every 100 iterations
            if (self.iter + 1) % 100 == 0:
                print(f"Current loss weights at iter {self.iter}: {self.model.criterion.weight_dict}")



    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
