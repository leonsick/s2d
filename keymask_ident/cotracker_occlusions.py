import sys
import warnings
from base64 import b64encode
import os, glob, json

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from crw_utils import load_image_robust





def load_masks(mask_folder: str) -> torch.Tensor:
    """
    Load an ordered sequence of multi‐color PNG masks from a folder and
    convert them into a single mask‐ID tensor.

    Args:
        mask_folder: Path to a directory containing one .png mask per frame.
                     Other files are ignored.

    Returns:
        masks: torch.Tensor of shape (T, H, W, 1), dtype torch.long,
               where each pixel is 0 for background or 1..N for each color‐mask.
    """
    # 1) collect all .png paths
    paths = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
    if not paths:
        warnings.warn(f"No .png masks found in {mask_folder!r}")
        return None

    id_maps = []
    for p in paths:
        # 2) read as RGB
        # Try to load the image with OpenCV first
        # bgr = cv2.imread(p)
        bgr = load_image_robust(p)

        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        H, W, _ = rgb.shape

        # 3) find all unique colors in this frame
        pixels = rgb.reshape(-1, 3)
        uniq = np.unique(pixels, axis=0)
        # filter out black
        colors = [tuple(c) for c in uniq if not np.all(c == 0)]

        # 4) build a color→ID map (black→0, others 1..)
        color2id = {(0, 0, 0): 0}
        for idx, col in enumerate(sorted(colors), start=1):
            color2id[col] = idx

        # 5) create the ID map for this frame
        id_map = np.zeros((H, W), dtype=np.int64)
        for col, idx in color2id.items():
            if idx == 0:
                continue
            mask = np.all(rgb == col, axis=2)
            id_map[mask] = idx

        id_maps.append(id_map[..., None])  # (H, W, 1)

    if not id_maps:
        raise RuntimeError("No valid mask images could be read.")

    if len(id_maps) == 0:
        warnings.warn("No valid mask images could be read.")
        return None

    # 6) stack into (T, H, W, 1) and convert to tensor
    masks_np = np.stack(id_maps, axis=0)
    return torch.from_numpy(masks_np)  # dtype=torch.long


def mp4_from_images(img_folder: str, frame_rate: int = 30) -> torch.Tensor:
    """
    Read all images in `img_folder` (sorted by filename), stack into a video tensor,
    and return a torch.Tensor of shape (1, T, C, H, W), dtype float.

    Args:
        img_folder: Path to your folder of image frames.
        frame_rate: Frames per second (unused here, kept for API consistency).

    Returns:
        video: torch.Tensor with shape (1, T, C, H, W), dtype float.
    """
    # collect image files
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_folder, e)))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {img_folder!r}")

    # read and convert to RGB
    frames = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    if not frames:
        raise ValueError("None of the images could be read successfully.")

    # stack into (T, H, W, C)
    video_np = np.stack(frames, axis=0)  # uint8

    # convert to torch.Tensor and permute to (1, T, C, H, W)
    video_t = (
        torch.from_numpy(video_np)
        .permute(0, 3, 1, 2)[None]
        .float()
    )
    return video_t


def get_segmentation_mask(masks: torch.Tensor,
                          query_frame_idx: int,
                          object_id: int = 1) -> torch.Tensor:
    """
    Extract a mask for one object (or all objects) from a (T, H, W, 1) long‐tensor.

    Args:
        masks:            torch.Tensor of shape (T, H, W, 1), dtype long,
                          where each pixel ∈ {0=bg, 1,2,…}.
        query_frame_idx:  index in [0..T-1] of the frame you want.
        object_id:        which mask ID to extract (default=1). If -1,
                          returns a mask of all non-background IDs.

    Returns:
        segm_mask: torch.Tensor of shape (1, 1, H, W), dtype uint8,
                   with values 255 for the selected mask region(s), 0 elsewhere.
    """
    # pull out the H×W slice
    frame_ids: torch.Tensor = masks[query_frame_idx, ..., 0]  # shape: (H, W)

    # build binary mask: either one object or all non-bg
    if object_id == -1:
        sel = frame_ids != 0
    else:
        sel = frame_ids == object_id

    segm_mask = sel.to(torch.uint8) * 255  # (H, W) in {0,255}

    # add batch and channel dims → (1, 1, H, W)
    return segm_mask.unsqueeze(0).unsqueeze(0)



def extract_appearance_events(
    vis: torch.Tensor,
    smoothing_window: int = 1,
    thresh: float = 0.95,
    min_run_length: int = 4
) -> dict[int, list[tuple[int, int]]]:
    """
    Args:
        vis:            (N, T) tensor, float in [0,1]: visibility curves.
        smoothing_window:   odd int >= 1, length of moving‐avg filter.
        thresh:         float in (0,1), threshold to binarize visibility.
        min_run_length: int >= 1, minimum length (in frames) of visible/invisible runs to keep.

    Returns:
        events: dict mapping object index → list of (appear_frame, disappear_frame) pairs.
    """

    N, T = vis.shape
    device = vis.device

    # 1) Smooth the curves to suppress single‐frame noise.
    #    Equivalent to a 1D moving average of length `smoothing_window`.
    pad = (smoothing_window - 1) // 2
    # vis.unsqueeze(1): (N,1,T) so conv1d treats T as “width”
    vis_padded = F.pad(vis.unsqueeze(1), (pad, pad), mode='reflect')
    kernel = torch.ones(1, 1, smoothing_window, device=device) / smoothing_window
    vis_smooth = F.conv1d(vis_padded, kernel, stride=1).squeeze(1)  # → (N, T)

    # 2) Threshold into a crisp binary signal: above thresh = visible.
    vis_bin = (vis_smooth >= thresh).to(vis.dtype)  # still (N, T), values {0.0,1.0}

    # 3) Morphological opening (erosion then dilation) to remove short flickers:
    #    - Erosion with kernel `min_run_length` removes any run of 1s shorter than that.
    #    - Dilation restores shape of longer runs.
    k = min_run_length
    pad2 = (k - 1) // 2

    # 3a) Erosion = min over window.  We get min by inverting and doing max‐pool:
    inv = 1.0 - vis_bin  # flips 0↔1
    inv_padded = F.pad(inv.unsqueeze(1), (pad2, pad2), mode='reflect')
    eroded = 1.0 - F.max_pool1d(inv_padded, kernel_size=k, stride=1).squeeze(1)

    # 3b) Dilation = max over window:
    eroded_padded = F.pad(eroded.unsqueeze(1), (pad2, pad2), mode='reflect')
    opened = F.max_pool1d(eroded_padded, kernel_size=k, stride=1).squeeze(1)  # (N, T)

    # 4) Find transitions 0→1 (appear) and 1→0 (disappear)
    diffs = opened[:, 1:] - opened[:, :-1]  # shape (N, T-1)
    events: dict[int, list[tuple[int,int]]] = {}
    for obj_idx in range(N):
        d = diffs[obj_idx]
        # where d==1 means frame i→i+1 went OFF→ON, so appear at i+1
        starts = (d == 1).nonzero(as_tuple=False).squeeze(1) + 1
        # where d==-1 means ON→OFF, so disappear at i+1
        ends   = (d == -1).nonzero(as_tuple=False).squeeze(1) + 1
        events[obj_idx] = list(zip(starts.tolist(), ends.tolist()))

    return events


def boolean_visibility(
    vis: torch.Tensor,
    threshold: float = 0.3
) -> torch.Tensor:
    """
    Convert visibility tensor to boolean tensor based on a threshold.

    Args:
        vis:        (N, T) tensor, float in [0,1]: visibility curves.
        threshold:  float in (0,1), threshold to binarize visibility.

    Returns:
        vis_bool:   (N, T) tensor, boolean visibility.
    """
    return vis >= threshold


def extract_object_visibility_data(video_path, masks_path, video_output_dir, visibility_maps_base_output_dir, debug=False):
    """
    Extract visibility data for each object in the video frames.

    Args:
        video_paths: Path to video JPEGs
        mask_paths:  Path to pseudo mask directory

    Returns:
        visibility_data: Dictionary containing visibility data for each object.
    """
    print("Predicting visibility maps for video: ", video_path)

    print("Loading masks...")
    masks = load_masks(masks_path)

    if masks is None:
        print("Failed to load masks for visibility analysis...")
        return None

    print("Masks loaded.")

    if video_path.__contains__("DAVIS"):
        dataset_name = "DAVIS"
    elif video_path.__contains__("ytvis2021"):
        dataset_name = "ytvis2021"
    elif video_path.__contains__("ytvis2019"):
        dataset_name = "ytvis2019"
    elif video_path.__contains__("ovis"):
        dataset_name = "ovis"
    elif video_path.__contains__("VIPSeg"):
        dataset_name = "VIPSeg"
    elif video_path.__contains__("MOSE"):
        dataset_name = "MOSE"
    elif video_path.__contains__("sa-v"):
        dataset_name = "SA-V"
    else:
        raise ValueError("Unknown dataset")

    if video_path.__contains__("train"):
        split = "train"
    elif video_path.__contains__("valid"):
        split = "valid"
    elif video_path.__contains__("test"):
        split = "test"
    elif video_path.__contains__("val"):
        split = "val"
    elif video_path.__contains__("imgs"):
        split = "imgs"
    else:
        split = "all"

    visibility_maps_output_dir = os.path.join(visibility_maps_base_output_dir, dataset_name, split)

    if not os.path.exists(visibility_maps_output_dir):
        try:
            os.makedirs(visibility_maps_output_dir)
        except Exception as e:
            print(f"Failed to create visibility maps output directory {visibility_maps_output_dir}: {e}")
            return None

    video_name = os.path.basename(video_path)
    print("Loading video...")
    video = mp4_from_images(video_path)
    print("Video loaded.")

    if os.path.exists("/mnt/data/checkpoints"):
        ckpt_path = "/mnt/data/checkpoints"
    elif os.path.exists("/mnt/hdd/leon/checkpoints/"):
        ckpt_path = "/mnt/hdd/leon/checkpoints/"
    else:
        raise FileNotFoundError(
            "Checkpoint directory not found. Please ensure the path is correct. Add it for the new cluster")

    print("Initializing model...") if debug else None
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            ckpt_path,
            'scaled_offline.pth'
        )
    )
    print("Model initialized.") if debug else None

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()

    video_visibilities = []
    video_visibilities_data = []
    for grid_query_frame in tqdm(range(video.shape[1]), desc="Gathering visibility data for each mask"):
        backward_tracking = grid_query_frame > 0
        forward_tracking = grid_query_frame < video.shape[1] - 1
        grid_size = 50

        frame_ids = torch.sort(torch.unique(masks[grid_query_frame, ..., 0])[1:])[
            0]  # Frame ids without 0 (background)

        if len(frame_ids) == 0:
            print(f"Frame {grid_query_frame} has no objects, skipping...")
            continue
        else:
            frame_visibilities = []
            frame_visibilities_data = []

        for oid in frame_ids:
            filename = f"frame_{grid_query_frame}_mask_{oid}_grid_{grid_size}"
            segm_mask = get_segmentation_mask(masks, grid_query_frame, object_id=oid)

            if torch.sum(segm_mask) == 0:
                print(f"Frame {grid_query_frame} has no objects, skipping...")
                continue

            pred_tracks, pred_visibility = model(video, grid_size=grid_size, grid_query_frame=grid_query_frame,
                                                 segm_mask=segm_mask, backward_tracking=backward_tracking)

            # Avg. Visibility per Frame
            frame_visibility = torch.mean(pred_visibility.float(), dim=2)
            frame_visibilities.append(frame_visibility.squeeze(0))

            # Save the visibility data for this object
            frame_visibilities_data.append(
                {"object_id": oid.item(), "visibility": frame_visibility.squeeze(0).tolist()})

            save_video = False

            if save_video:
                vis = Visualizer(save_dir=os.path.join(video_output_dir, dataset_name, split, video_name),
                                 pad_value=100)
                try:
                    vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename=filename)
                except Exception as e:
                    print(f"Error visualizing video {video_name} with filename {filename}: {e}")
                    continue

        video_visibilities.append(torch.stack(frame_visibilities))

        video_visibilities_data.append({
            "frame_id": grid_query_frame,
            "data": frame_visibilities_data
        })

    if len(video_visibilities) == 0:
        return None

    # Save the visibility data to a JSON file
    visibility_data_path = os.path.join(visibility_maps_output_dir, "data", video_name + ".json")
    if not os.path.exists(os.path.dirname(visibility_data_path)):
        os.makedirs(os.path.dirname(visibility_data_path))

    with open(visibility_data_path, 'w') as f:
        json.dump({"video_data": video_visibilities_data}, f, indent=4)
    print(f"Visibility data saved to {visibility_data_path}") if debug else None

    return {"video_data": video_visibilities_data}

