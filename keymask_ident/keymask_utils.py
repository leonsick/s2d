from __future__ import print_function

import os
import sys
import time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import crw_utils


def extract_visibility_data(visibility_data):
    extracted_data = []
    clusters = visibility_data['clusters']
    video_name = visibility_data['video_name']

    for cluster in clusters:
        cluster_data = []
        all_candidates = cluster['all_candidates']
        for candidate in all_candidates:
            cluster_data.append({
                'range': candidate['range'],
                'mask_candidates': candidate['candidates']
            })

        extracted_data.append(cluster_data)
    return extracted_data, video_name


def get_segmentation_mask(masks: torch.Tensor,
                          query_frame_idx: int,
                          object_id: int = 1) -> torch.Tensor:
    """
    Extract a mask for one object (or all objects) from a (T, 1, H, W) tensor.

    Args:
        masks:            torch.Tensor of shape (T, 1, H, W), dtype long,
                          where each pixel ∈ {0=bg, 1,2,…}.
        query_frame_idx:  index in [0..T-1] of the frame you want.
        object_id:        which mask ID to extract (default=1). If -1,
                          returns a mask of all non-background IDs.

    Returns:
        segm_mask: torch.Tensor of shape (1, 1, H, W), dtype uint8,
                   with values 255 for the selected mask region(s), 0 elsewhere.
    """
    # pull out the H×W slice from the channel dimension
    masks = masks.permute(0, 2, 3, 1)  # shape: (T, H, W, 1)
    frame_ids: torch.Tensor = masks[query_frame_idx, ..., 0]  # shape: (H, W)

    # build binary mask: either one object or all non-bg
    if object_id == -1:
        sel = frame_ids != 0
    else:
        sel = frame_ids == object_id

    segm_mask = sel.to(torch.uint8) * 255  # (H, W) in {0,255}

    # add batch and channel dims → (1, 1, H, W)
    return segm_mask.unsqueeze(0).unsqueeze(0)


def save_segmentation_masks(imgs, imgs_orig, lbls, meta, save_dir, debug=False):
    N = imgs.shape[0]


    visibility_data, video_name = extract_visibility_data(meta['visibility'])
    video_data_save_dir = os.path.join(save_dir, video_name)

    if not os.path.exists(video_data_save_dir):
        os.makedirs(video_data_save_dir, exist_ok=True)


    print(f"Found {len(visibility_data)} visibility clusters in video {video_name}.") if debug else None

    for cluster_id, cluster_data in enumerate(visibility_data):
        print(f"Starting with Cluster {cluster_id} with ranges {[d['range'] for d in cluster_data]}.") if debug else None

        for range_data in cluster_data:
            start_frame = range_data['range'][0]
            end_frame = range_data['range'][1]

            imgs_range = imgs[start_frame:end_frame + 1]
            imgs_orig_range = imgs_orig[start_frame:end_frame + 1]
            lbls_range = lbls[start_frame:end_frame + 1]

            print(imgs_range.shape, imgs_orig_range.shape, lbls_range.shape)

            print(f"Processing range {start_frame} to {end_frame}.") if debug else None

            # Extract the mask candidates
            mask_candidates = range_data['mask_candidates']

            # Process the mask candidates
            for candidate in mask_candidates:
                frame_id = candidate['frame_id']
                mask_id = candidate['mask_id']
                print(f"Processing candidate with frame ID {frame_id} and mask ID {mask_id}.") if debug else None

                # Get the binary mask
                segm_mask = get_segmentation_mask(lbls, frame_id, object_id=mask_id)

                print(f"Segmentation mask shape: {segm_mask.shape}") if debug else None

                # segm_mask: torch.Tensor of shape (1,1,H,W), dtype uint8
                # Move to CPU & numpy, then squeeze out the 1’s → (H, W)
                mask = segm_mask.squeeze().cpu().numpy()

                cluster_save_dir = os.path.join(video_data_save_dir, f"cluster_{cluster_id}")

                if not os.path.exists(cluster_save_dir):
                    os.makedirs(cluster_save_dir, exist_ok=True)

                # build filename
                filename = f"cluster{cluster_id}_frame{frame_id}_mask{mask_id}.png"
                out_path = os.path.join(cluster_save_dir, filename)

                # save via PIL
                Image.fromarray(mask).save(out_path)

    return video_data_save_dir
