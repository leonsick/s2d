# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
#import .transf
from fvcore.transforms import BlendTransform, TransformList

from .augmentation import build_augmentation

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target

class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """

def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            expected_wh = (dataset_dict["height"], dataset_dict["width"])
            dataset_dict["height"], dataset_dict["width"] = dataset_dict["width"], dataset_dict["height"]
            if image_wh != expected_wh:
                raise SizeMismatchError(
                    "Mismatched image shape{}, got {}, expect {}.".format(
                        " for image " + dataset_dict["file_name"]
                        if "file_name" in dataset_dict
                        else "",
                        image_wh,
                        expected_wh,
                    )
                    + " Please check the width/height in your annotation."
                )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]

class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        return_clean_image: bool = False,
        dense_annotation_selection: bool = True,
        disentangle_distillation_loader: bool = False,
        distillation_dense_annotation_selection: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.return_clean_image     = return_clean_image
        self.dense_annotation_selection = dense_annotation_selection
        self.disentangle_distillation_loader = disentangle_distillation_loader
        self.distillation_dense_annotation_selection = distillation_dense_annotation_selection
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        return_clean_image = cfg.MODEL.META_ARCHITECTURE == "KDVideoMaskFormer"

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "return_clean_image": return_clean_image,
            "dense_annotation_selection": cfg.INPUT.DENSE_ANNOTATION_SELECTION,
            "disentangle_distillation_loader": cfg.INPUT.DISENTANGLE_DISTILLATION_LOADER,
            "distillation_dense_annotation_selection": cfg.INPUT.DISTILLATION_DENSE_ANNOTATION_SELECTION
        }

        return ret

    def dense_frame_selection(self, video_annos, video_length):
        instance_tracks = {}
        for frame_idx, annos in enumerate(video_annos):
            for anno in annos:
                inst_id = anno["id"]
                if inst_id not in instance_tracks:
                    instance_tracks[inst_id] = []
                instance_tracks[inst_id].append(frame_idx)

        possible_windows = []
        for inst_id, frames in instance_tracks.items():
            if len(frames) < self.sampling_frame_num:
                continue

            # Find consecutive frame sequences for this instance
            for i in range(len(frames) - self.sampling_frame_num + 1):
                # Check if the frames are consecutive
                is_consecutive = True
                for j in range(self.sampling_frame_num - 1):
                    if frames[i + j + 1] != frames[i + j] + 1:
                        is_consecutive = False
                        break

                if is_consecutive:
                    start_frame = frames[i]
                    possible_windows.append(list(range(start_frame, start_frame + self.sampling_frame_num)))

        if possible_windows:
            # If there are such windows, pick one randomly
            selected_idx = random.choice(possible_windows)

        else:
            # Fallback to the original sparse sampling logic if no dense window is found
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame - self.sampling_frame_range)
            end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

            valid_indices = list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))
            # Ensure we don't sample more frames than available
            num_to_sample = min(self.sampling_frame_num - 1, len(valid_indices))

            selected_idx = np.random.choice(
                np.array(valid_indices),
                num_to_sample,
                replace=False
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)

        return selected_idx

    def random_frame_selection(self, video_length):
        ref_frame = random.randrange(video_length)

        start_idx = max(0, ref_frame - self.sampling_frame_range)
        end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

        selected_idx = np.random.choice(
            np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
            self.sampling_frame_num - 1,
        )
        selected_idx = selected_idx.tolist() + [ref_frame]
        selected_idx = sorted(selected_idx)
        if self.sampling_frame_shuffle:
            random.shuffle(selected_idx)

        return selected_idx

    def gather_dataset_dict(self, dataset_dict, video_annos, file_names, selected_idx, distillation_data=False):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Get the annotations for indicies
        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        dataset_dict["clean_image"] = [] if self.return_clean_image and self.is_train else None
        dataset_dict["transforms_matrix"] = [] if self.return_clean_image and self.is_train else None
        dataset_dict["transforms"] = [] if self.return_clean_image and self.is_train else None

        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            check_image_size(dataset_dict, image)

            if self.return_clean_image and self.is_train:
                dataset_dict["clean_image"].append(
                    torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                )



                orig_h, orig_w = dataset_dict["clean_image"][-1].shape[1], dataset_dict["clean_image"][-1].shape[2]

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            if self.return_clean_image and self.is_train:
                idx_hw = torch.arange(orig_h * orig_w, device=dataset_dict["clean_image"][-1].device,
                                      dtype=torch.float32).view(orig_h, orig_w)

                transforms_matrix = transforms.apply_segmentation(idx_hw.numpy())
                assert aug_input.image.shape[:2] == transforms_matrix.shape[
                                                    :2], "Augmentation transforms should not change the image size, but got {} and {}".format(
                    aug_input.image.shape[:2], transforms_matrix.shape[:2]
                )

                dataset_dict["transforms_matrix"].append(
                    torch.as_tensor(transforms_matrix, dtype=torch.long).to(dataset_dict["clean_image"][-1].device))

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]

            if self.return_clean_image and self.is_train:
                new_transforms = []
                for t in transforms:
                    if not isinstance(t, BlendTransform):
                        new_transforms.append(t)

                transforms = TransformList(new_transforms)

                # print("transforms", transforms)

                dataset_dict["transforms"].append(transforms)

            # print("self.num_classes", self.num_classes)
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]
            #sorted_annos = [_get_dummy_anno(0) for _ in range(len(ids))]

            for _anno in annos:
                #_anno["category_id"] = 0  # single class
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            # print all category ids of sorted_annos
            #print("Category IDs in sorted_annos:", [anno["category_id"] for anno in sorted_annos])
            # print the sum of each segmentation mask in sorted_annos
            #print("Sum of segmentation masks in sorted_annos:", [np.sum(anno["segmentation"]) for anno in sorted_annos])

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        if distillation_data:
            # Rename the keys for teacher data
            dataset_dict["distill_image"] = dataset_dict.pop("image")
            dataset_dict["distill_instances"] = dataset_dict.pop("instances")
            dataset_dict["distill_file_names"] = dataset_dict.pop("file_names")
            if self.return_clean_image and self.is_train:
                dataset_dict["distill_clean_image"] = dataset_dict.pop("clean_image")
                dataset_dict["distill_transforms_matrix"] = dataset_dict.pop("transforms_matrix")
                dataset_dict["distill_transforms"] = dataset_dict.pop("transforms")


        # Across all instances from the dataset dict, print the per-frame category ids an the sum of each segmentation mask
        """if self.is_train:
            print("--- Dataset Dict Info ---")
            print("Selected frame indices:", selected_idx)

            for frame_idx, inst in zip(selected_idx, dataset_dict["instances"]):
                if len(inst) > 0:
                    print(f"Frame {frame_idx}: Category IDs:", inst.gt_classes.tolist())
                    if inst.has("gt_masks"):
                        print(f"Frame {frame_idx}: Sum of segmentation masks:", [mask.sum().item() for mask in inst.gt_masks])
                    else:
                        print(f"Frame {frame_idx}: No gt_masks available.")
                else:
                    print(f"Frame {frame_idx}: No instances available.")"""

        return dataset_dict

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below


        video_length = dataset_dict["length"]
        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            if self.dense_annotation_selection:
                # For each instance, find its consecutive frame sequences
                selected_idx = self.dense_frame_selection(video_annos, video_length)
            else:
                selected_idx = self.random_frame_selection(video_length)
        else:
            selected_idx = range(video_length)

        dataset_dict = self.gather_dataset_dict(dataset_dict, video_annos, file_names, selected_idx)

        if self.is_train and self.disentangle_distillation_loader:
            # Create a second set of images and annotations for the teacher model
            # with different random frame selection
            if self.distillation_dense_annotation_selection:
                distillation_selected_idx = self.dense_frame_selection(video_annos, video_length)
            else:
                distillation_selected_idx = self.random_frame_selection(video_length)

            distillation_dataset_dict = self.gather_dataset_dict(dataset_dict, video_annos, file_names, distillation_selected_idx, distillation_data=True)

            # Combine the dicts
            dataset_dict.update(distillation_dataset_dict)

        return dataset_dict


import torch


def apply_transformation_frame_by_frame(original_predictions, batched_indices, gt_masks_per_video):
    """
    Applies a transformation to a sequence of images frame by frame.

    Args:
        original_predictions (torch.Tensor): A tensor of original images of shape (P, T, H, W).
        batched_indices (list): A list of T tensors, each potentially of different shape.
        gt_masks_per_video (torch.Tensor): The target tensor to write to, shape (P, T, H_out, W_out).

    Returns:
        torch.Tensor: The modified gt_masks_per_video tensor.
    """
    P, T, H, W = original_predictions.shape
    H_out, W_out = gt_masks_per_video.shape[-2:]

    print_once = True  # Flag to ensure we print the warning only once

    # Iterate through the predictions (P) and time steps (T)
    for p in range(P):
        for t in range(T):
            original_img = original_predictions[p, t, :, :]
            transformed_indices = batched_indices[t].long()  # Ensure indices are of type long

            original_img_flat = original_img.view(-1)

            # Use torch.take to apply the transformation
            # Ensure indices are on the same device as the data
            new_img = torch.take(original_img_flat, transformed_indices.to(original_img_flat.device))

            # Cast new_img to a floating-point type before interpolation
            # This is the key fix for the "Bool" error
            if new_img.dtype == torch.bool:
                new_img = new_img.float()

            # Resize new_img to match the destination tensor's shape
            # if new_img.shape[0] != H_out or new_img.shape[1] != W_out: This is the old, erroneous condition
            if new_img.shape[0] >= H_out or new_img.shape[1] >= W_out:
                # Reshape for resizing (1, 1, H_new, W_new)
                if print_once:
                    print("Resizing new_img from shape", new_img.shape, "to", (H_out, W_out))
                    print_once = False

                new_img = new_img.unsqueeze(0).unsqueeze(0)
                new_img = torch.nn.functional.interpolate(
                    new_img,
                    size=(H_out, W_out),
                    mode='bilinear',
                    align_corners=False
                )
                new_img = new_img.squeeze() > 0.  # Back to (H_out, W_out)

            new_h, new_w = new_img.shape
            # Place the transformed image in the correct location
            gt_masks_per_video[p, t, :new_h, :new_w] = new_img

    return gt_masks_per_video


def apply_transformslist_frame_by_frame(original_predictions, batched_indices, gt_masks_per_video):
    """
    Applies a transformation to a sequence of images frame by frame.

    Args:
        original_predictions (torch.Tensor): A tensor of original images of shape (P, T, H, W).
        batched_indices (list): A list of T TransformList objects, each potentially of different shape.
        gt_masks_per_video (torch.Tensor): The target tensor to write to, shape (P, T, H_out, W_out).

    Returns:
        torch.Tensor: The modified gt_masks_per_video tensor.
    """
    P, T, H, W = original_predictions.shape
    H_out, W_out = gt_masks_per_video.shape[-2:]

    print_once = True  # Flag to ensure we print the warning only once

    # Iterate through the predictions (P) and time steps (T)
    for p in range(P):
        for t in range(T):
            original_img = original_predictions[p, t, :, :]
            transforms = batched_indices[t]  # This is now a TransformList object

            # Apply the transformations using the apply_image method
            new_img = transforms.apply_image(original_img.cpu().numpy())
            new_img = torch.from_numpy(new_img).to(original_img.device)

            # Cast new_img to a floating-point type before interpolation
            # This is the key fix for the "Bool" error
            if new_img.dtype == torch.bool:
                new_img = new_img.float()

            # Resize new_img to match the destination tensor's shape
            # if new_img.shape[0] != H_out or new_img.shape[1] != W_out: This is the old, erroneous condition
            if new_img.shape[0] >= H_out or new_img.shape[1] >= W_out:
                # Reshape for resizing (1, 1, H_new, W_new)
                if print_once:
                    print("Resizing new_img from shape", new_img.shape, "to", (H_out, W_out))
                    print_once = False

                new_img = new_img.unsqueeze(0).unsqueeze(0)
                new_img = torch.nn.functional.interpolate(
                    new_img,
                    size=(H_out, W_out),
                    mode='bilinear',
                    align_corners=False
                )
                new_img = new_img.squeeze() > 0.  # Back to (H_out, W_out)

            new_h, new_w = new_img.shape
            # Place the transformed image in the correct location
            gt_masks_per_video[p, t, :new_h, :new_w] = new_img
    return gt_masks_per_video


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict
