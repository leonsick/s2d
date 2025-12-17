# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/Mask2Former/tree/main/mask2former_video


import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
import tqdm
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

#from videocutler.cutler.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


YTVIS_CATEGORIES_2019 = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"},
]


IMAGENET_CATEGORIES_cls_agnostic = [
    # {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "fg"},
    # {"color": [120, 166, 157], "isthing": 1, "id": 1, "name": "fg"},
    {"color": [73, 77, 174], "isthing": 1, "id": 1, "name": "fg"},
    # {"color": [199, 100, 0], "isthing": 1, "id": 2, "name": "fg"},
]

DAVIS_CATEGORIES_cls_agnostic = [
    # {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "fg"},
    #{"color": [120, 166, 157], "isthing": 1, "id": 2, "name": "no_object"},
    {"color": [73, 77, 174], "isthing": 1, "id": 1, "name": "fg"},
    # {"color": [199, 100, 0], "isthing": 1, "id": 2, "name": "fg"},
]

#['Person', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Poultry', 'Giant_panda', 'Lizard', 'Parrot', 'Monkey', 'Rabbit', 'Tiger', 'Fish', 'Turtle', 'Bicycle', 'Motorcycle', 'Airplane', 'Boat', 'Vehical']

OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "Sheep"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "Poultry"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "Fish"},        # Replaced 'Bicycle' with 'Fish'
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "Bicycle"},     # Shifted from ID 19
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "Motorcycle"},  # Shifted from ID 21
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "Airplane"}, # Shifted from ID 22
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "Boat"},        # Shifted from ID 23
    {"color": [0, 20, 0], "isthing": 1, "id": 25, "name": "Vehical"},         # New entry to match the list length (color is arbitrary)
]

CITYSCAPES_CATEGORIES = [
    {'id': 11, 'name': 'person', 'supercategory': 'human', 'isthing': 1, 'instance_eval': 1, 'trainid': 11, 'ori_id': 24, 'color': [220, 20, 60]},
    {'id': 12, 'name': 'rider', 'supercategory': 'human', 'isthing': 1, 'instance_eval': 1, 'trainid': 12, 'ori_id': 25, 'color': [255, 0, 0]},
    {'id': 13, 'name': 'car', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 13, 'ori_id': 26, 'color': [0, 0, 142]},
    {'id': 14, 'name': 'truck', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 14, 'ori_id': 27, 'color': [0, 0, 70]},
    {'id': 15, 'name': 'bus', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 15, 'ori_id': 28, 'color': [0, 60, 100]},
    {'id': 16, 'name': 'train', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 16, 'ori_id': 31, 'color': [0, 80, 100]},
    {'id': 17, 'name': 'motorcycle', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 17, 'ori_id': 32, 'color': [0, 0, 230]},
    {'id': 18, 'name': 'bicycle', 'supercategory': 'vehicle', 'isthing': 1, 'instance_eval': 1, 'trainid': 18, 'ori_id': 33, 'color': [119, 11, 32]}
]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


def _get_ytvis_2019_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_ytvis_2021_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_cityscapes_instances_meta():
    thing_ids = [k["id"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    #assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_imagenet_cls_agnostic_instances_meta():
    thing_ids = [k["id"] for k in IMAGENET_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    thing_colors = [k["color"] for k in IMAGENET_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in IMAGENET_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_davis_cls_agnostic_instances_meta():
    thing_ids = [k["id"] for k in DAVIS_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DAVIS_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DAVIS_CATEGORIES_cls_agnostic if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    try:
        from .ytvis_api.ytvos import YTVOS
    except ImportError:
        print("Failed to import YTVOS from ytvis_api. Retrying.")
        from ytvis_api.ytvos import YTVOS


    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        print("Thing classes", thing_classes)
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )




if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    #from detectron2.utils.visualizer import Visualizer
    from custom_visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("mose_dense8_wclass_nms01_cls_agnostic")
    #meta = MetadataCatalog.get("ytvis_2021_train_dense6_nms_cls_agnostic")
    #meta = MetadataCatalog.get("vipseg_dense8_wclass_nms01_cls_agnostic")

    json_file = "/mnt/hdd/leon/shared_datasets/MOSE/mose_wclass_nms01_converted_score_filtered8.json"
    image_root = "/mnt/hdd/leon/shared_datasets/MOSE/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="mose_dense8_wclass_nms01_cls_agnostic")

    #json_file = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/converted_score_filtered6_nms.json"
    #image_root = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/JPEGImages"
    #dicts = load_ytvis_json(json_file, image_root, dataset_name="mose_dense8_nms025_cls_agnostic")

    #json_file = "/mnt/hdd/leon/shared_datasets/VIPSeg/vipseg_wclass_nms01_converted_score_filtered8.json"
    #image_root = "/mnt/hdd/leon/shared_datasets/VIPSeg/imgs"
    #dicts = load_ytvis_json(json_file, image_root, dataset_name="vipseg_dense8_wclass_nms01_cls_agnostic")

    # Visualization: Sparse-to-Dense
    # Sparse
    #meta = MetadataCatalog.get("ytvis_2021_train_sparse_2829_cls_agnostic")
    #json_file = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/merged_2829anns.json"
    #image_root = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/JPEGImages"
    #dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2021_train_sparse_2829_cls_agnostic")
    #dirname = "/mnt/hdd/leon/outputs/video-data-vis/ytvis2021/ytvis_2021_train_sparse_2829_cls_agnostic/"

    # Dense
    meta = MetadataCatalog.get("ytvis_2021_train_dense6_nms_cls_agnostic")
    json_file = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/converted_score_filtered6_nms.json"
    image_root = "/mnt/hdd/leon/shared_datasets/ytvis2021/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2021_train_dense6_nms_cls_agnostic")
    dirname = "/mnt/hdd/leon/outputs/video-data-vis/ytvis2021/ytvis_2021_train_dense6_nms_cls_agnostic/"


    logger.info("Done loading {} samples.".format(len(dicts)))

    #dirname = "/mnt/hdd/leon/outputs/video-data-vis/MOSE/mose_dense8_wclass_nms01_cls_agnostic/"
    #dirname = "/mnt/hdd/leon/outputs/video-data-vis/ytvis2021/ytvis_2021_train_dense6_nms_cls_agnostic/"
    #dirname = "/mnt/hdd/leon/outputs/video-data-vis/VIPSeg/vipseg_dense8_wclass_nms01_cls_agnostic/"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in tqdm.tqdm(dicts):
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_video_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
