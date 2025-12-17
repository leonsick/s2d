# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/Mask2Former/tree/main/mask2former_video

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_imagenet_cls_agnostic_instances_meta,
    _get_davis_cls_agnostic_instances_meta,
    _get_ovis_instances_meta,
    _get_cityscapes_instances_meta
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis2019/train/JPEGImages",
                         "ytvis2019/instances_train_sub.json"),
    "ytvis_2019_val": ("ytvis2019/valid/JPEGImages",
                       "ytvis2019/instances_val_sub.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis2021/train/JPEGImages",
                         "ytvis2021/train/instances.json"),
    "ytvis_2021_valid": ("ytvis2021/valid/JPEGImages",
                       "ytvis2021/valid/valid_gt.json"),
    "ytvis_2021_test": ("ytvis2021/test/JPEGImages",
                        "ytvis2021/test/instances.json"),
    "ytvis_2021_minus_2019_train": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_val_sub.json"),
}

_PREDEFINED_SPLITS_YTVIS_2022 = {
    "ytvis_2022_train": ("ytvis2021/train/JPEGImages",
                         "ytvis2021/train/instances.json"),
    "ytvis_2022_valid": ("ytvis2022/valid/JPEGImages",
                       "ytvis2022/annotations/gt.json"),
    "ytvis_2022_valid_short": ("ytvis2022/valid/JPEGImages",
                       "ytvis2022/annotations/gt_short.json"),
    "ytvis_2022_valid_long": ("ytvis2022/valid/JPEGImages",
                       "ytvis2022/annotations/gt_long.json"),
    "ytvis_2022_test": ("ytvis2021/test/JPEGImages",
                        "ytvis2021/test/instances.json"),
    "ytvis_2022_minus_2019_train": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_val_sub.json"),
}


_PREDEFINED_SPLITS_YTVIS_CLS_AGNOSTIC = {
    # YTVIS 2021
    "ytvis_2021_train_cls_agnostic": ("ytvis2021/train/JPEGImages",
                                    "ytvis2021/train/instances_cls_agnostic.json"),
    "ytvis_2021_train_dense_cls_agnostic": ("ytvis2021/train/JPEGImages",
                                    "ytvis2021/train/converted_annotations.json"),
}


_PREDEFINED_SPLITS_VIPSEG_CLS_AGNOSTIC = {
    "vipseg_cls_agnostic": ("VIPSeg/imgs",
                        "VIPSeg/VIPSeg_merged.json"),
}

_PREDEFINED_SPLITS_MOSE_CLS_AGNOSTIC = {
    "mose_cls_agnostic": ("MOSE/train/JPEGImages",
                        "MOSE/mose_merged.json"),
}

_PREDEFINED_SPLITS_SAV_CLS_AGNOSTIC = {
    "sa-v_cls_agnostic": ("sa-v/sav_train_jpeg",
                        "sa-v/sav_merged.json"),
}

def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_mose_cls_agnostic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOSE_CLS_AGNOSTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_imagenet_cls_agnostic_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_vipseg_cls_agnostic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIPSEG_CLS_AGNOSTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_imagenet_cls_agnostic_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_sav_cls_agnostic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SAV_CLS_AGNOSTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_imagenet_cls_agnostic_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvis_2022(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2022.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_cls_agnostic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_CLS_AGNOSTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_imagenet_cls_agnostic_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ytvis_cls_agnostic(_root)
    register_all_mose_cls_agnostic(_root)
    register_all_vipseg_cls_agnostic(_root)
    register_all_sav_cls_agnostic(_root)
    register_all_ytvis_2022(_root)