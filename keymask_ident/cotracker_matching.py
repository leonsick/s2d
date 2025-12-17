import shutil
import sys
import warnings
from base64 import b64encode
import os, glob, json
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from crw_utils import load_image_robust


def load_masks(mask_folder: str):
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
        #bgr = cv2.imread(p)
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


def load_cluster_masks(mask_folder: str):
    # Scan the directory for cluster folders, there is at least a cluster_0 folder
    cluster_folders_unfiltered = sorted(glob.glob(os.path.join(mask_folder, 'cluster_*')))
    cluster_folders = [folder for folder in cluster_folders_unfiltered if len(os.listdir(folder)) > 0]
    print(cluster_folders_unfiltered)
    print(cluster_folders)

    if not cluster_folders:
        warnings.warn(f"No cluster folders found in {mask_folder!r}. Skipping video!")
        return []

    # Iterate over the cluster folders and load the masks
    # The masks are simple binary masks, one mask per png
    # The name of the file contains meta info that we need to extract
    # Example: cluster1_frame0_mask0.png
    # Extract the cluster id, frame_id and mask_id from the filename
    all_clusters_and_masks = []
    for cluster_folder in cluster_folders:
        cluster_masks = []
        cluster_id = int(os.path.basename(cluster_folder).split('_')[1])
        mask_files = sorted(glob.glob(os.path.join(cluster_folder, '*.png')))
        for mask_file in mask_files:
            # Extract the frame_id and mask_id from the filename
            filename = os.path.basename(mask_file)
            parts = filename.split('_')
            frame_id = int(parts[1].replace('frame', ''))
            mask_id = int(parts[2].split('.')[0].replace('mask', ''))
            # Load the mask
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            # Convert to binary mask
            mask = (mask > 0).astype(np.uint8) * 255
            cluster_masks.append({
                "vis_cluster_id": cluster_id,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "mask": mask
            })
        all_clusters_and_masks.append(cluster_masks)

    return all_clusters_and_masks


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
    if query_frame_idx >= 0:
        frame_ids: torch.Tensor = masks[query_frame_idx, ..., 0]  # shape: (H, W)
    else:
        # if query_frame_idx is -1, we have the masks for one frame already
        frame_ids: torch.Tensor = masks[..., 0]

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
    events: dict[int, list[tuple[int, int]]] = {}
    for obj_idx in range(N):
        d = diffs[obj_idx]
        # where d==1 means frame i→i+1 went OFF→ON, so appear at i+1
        starts = (d == 1).nonzero(as_tuple=False).squeeze(1) + 1
        # where d==-1 means ON→OFF, so disappear at i+1
        ends = (d == -1).nonzero(as_tuple=False).squeeze(1) + 1
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


def contruct_frameid_maskid_lookup(all_video_masks) -> list[list[dict]]:
    global_counter = 0
    lookup_data = []
    for i_frame in range(all_video_masks.shape[0]):
        frame_data = []
        frame_oids = torch.sort(torch.unique(all_video_masks[i_frame, ..., 0])[1:])[
            0]  # Frame ids without 0 (background)
        for oid in frame_oids:
            frame_data.append({
                "frame_id": i_frame,
                "mask_id": oid.item(),
                "overall_mask_id": global_counter,
            })
            global_counter += 1

        lookup_data.append(frame_data)

    return lookup_data


def contruct_frameid_maskid_cluster_lookup(cluster_masks) -> list[list[dict]]:
    # Construct a lookup table to map frame_id and mask_id to a cluster_mask_id
    # The cluster mask id starts from 0 for every cluster
    lookup_data = []
    for cdata in cluster_masks:
        cluster_lookup = []
        cluster_counter = 0
        for cmask in cdata:
            cluster_lookup.append({
                "cluster_mask_id": cluster_counter,
                "frame_id": cmask["frame_id"],
                "mask_id": cmask["mask_id"],
            })
            cluster_counter += 1

        lookup_data.append(cluster_lookup)

    return lookup_data


def get_overall_maskid(frameid_maskid_lookup, frame_id, mask_id):
    """
    Get the overall mask ID for a given frame ID and mask ID.

    Args:
        frameid_maskid_lookup: List of lists containing frame_id, mask_id, and overall_mask_id.
        frame_id: Frame index.
        mask_id: Mask index.

    Returns:
        overall_mask_id: Overall mask ID for the given frame and mask.
    """
    for entry in frameid_maskid_lookup[frame_id]:
        if entry["mask_id"] == mask_id:
            return entry["overall_mask_id"]
    return None


def get_cluster_maskid(frameid_maskid_cluster_lookup, cluster_id, frame_id, mask_id):
    # Get the cluster mask ID for a given frame ID and mask ID.
    for entry in frameid_maskid_cluster_lookup[cluster_id]:
        if entry["frame_id"] == frame_id and entry["mask_id"] == mask_id:
            return entry["cluster_mask_id"]
    return None


def get_frameid_maskid_from_overall_maskid(frameid_maskid_lookup, overall_mask_id):
    """
    Get the frame ID and mask ID from the overall mask ID.

    Args:
        frameid_maskid_lookup: List of lists containing frame_id, mask_id, and overall_mask_id.
        overall_mask_id: Overall mask ID.

    Returns:
        A tuple (frame_id, mask_id) for the given overall mask ID.
    """
    for frame_data in frameid_maskid_lookup:
        for entry in frame_data:
            if entry["overall_mask_id"] == overall_mask_id:
                return entry["frame_id"], entry["mask_id"]

    raise ValueError(f"Overall mask ID {overall_mask_id} not found in lookup table.")


def load_visibility_data(visibility_maps_output_dir, video_name):
    """
    Load visibility data from the output directory.

    Args:
        visibility_maps_output_dir: Path to the output directory containing visibility maps.
        video_name: Name of the video.

    Returns:
        clusters_and_visibility_ranges: List of dictionaries containing cluster IDs and visibility ranges.
    """
    visibility_data_path = os.path.join(visibility_maps_output_dir, f"{video_name}.json")
    with open(visibility_data_path, 'r') as f:
        clusters_and_visibility_ranges = json.load(f)

    return clusters_and_visibility_ranges["clusters"]


def get_masks_for_vrange(masks, v_range):

    # This function should return the masks for the given visibility range
    filtered_masks = []
    for mask_data in masks:
        if mask_data["frame_id"] >= v_range[0] and mask_data["frame_id"] <= v_range[1]:
            filtered_masks.append(mask_data)
    return filtered_masks


def save_temporal_group_masks(mask_groupings, cluster_masks, visibility_group_mask_path, idx_correction=0):
    for c_maskgroup_data in mask_groupings:
        cid = c_maskgroup_data["cluster_id"]
        vis_cluster_path = os.path.join(visibility_group_mask_path, f"cluster_{cid}")
        print(f"Saving masks for cluster {cid} to {vis_cluster_path}")
        c_maskgroup = c_maskgroup_data["overall_mask_ids_per_label"]

        # Delete all folder that start with group to save the new results
        for folder in glob.glob(os.path.join(vis_cluster_path, 'group_*')):
            shutil.rmtree(folder)

        # print(c_maskgroup)
        for m_group, fid_mid_list in c_maskgroup.items():
            # Create a directory for the mask group
            mask_group_path = os.path.join(vis_cluster_path, f"group_{m_group}")
            os.makedirs(mask_group_path, exist_ok=True)
            vis_cluster_masks = cluster_masks[cid-idx_correction] if cid >= len(cluster_masks) else cluster_masks[cid]
            # print(fid_mid_list)
            for (fid, mid) in fid_mid_list:
                # print(fid)
                # print(mid)
                # Get the mask for the frame_id and mask_id
                mask_data = next((m for m in vis_cluster_masks if m['frame_id'] == fid and m['mask_id'] == mid), None)
                if mask_data is not None:
                    mask = mask_data['mask']
                    # Save the mask as a PNG file
                    filename = f"frame{fid}_mask{mid}.png"
                    filepath = os.path.join(mask_group_path, filename)

                    Image.fromarray(mask).save(filepath)


def save_cluster_coverages(video_coverage, cluster_coverages, visibility_to_temporal_factors, cluster_mask_path):
    video_coverage_path = os.path.join(cluster_mask_path, "video_coverage.txt")
    with open(video_coverage_path, 'w') as f:
        f.write(f"Video Coverage: {video_coverage:.2f}\n")

    actual_cids = sorted([
        int(d.split('_')[1]) for d in os.listdir(cluster_mask_path)
        if d.startswith('cluster_') and os.path.isdir(os.path.join(cluster_mask_path, d))
    ])

    for cid_idx, coverage in enumerate(cluster_coverages):
        cid = actual_cids[cid_idx]
        cluster_coverage_path = os.path.join(cluster_mask_path, f"cluster_{cid}", "cluster_coverage.txt")
        visibility_to_temporal_factor = visibility_to_temporal_factors[cid_idx]
        with open(cluster_coverage_path, 'w') as f:
            f.write(f"Cluster {cid} Coverage: {coverage:.2f}\n"
                    f"Visibility to Temporal Factor: {visibility_to_temporal_factor}\n")


def pred_tracks_to_binary_masks(pred_tracks: torch.Tensor, height: int, width: int,
                                return_mask: bool = False) -> torch.Tensor:
    """
    Convert predicted track points into per-frame binary masks by filling the convex polygon
    defined by the track points.

    Args:
        pred_tracks (torch.Tensor): Tensor of shape (B, T, P, 2) with point coordinates (x, y).
        height (int): Desired mask height (pixels).
        width (int): Desired mask width (pixels).

    Returns:
        torch.Tensor: Binary masks of shape (B, T, height, width), dtype=torch.uint8.
    """
    B, T, P, _ = pred_tracks.shape
    device = pred_tracks.device
    masks = torch.zeros((B, T, height, width), dtype=torch.uint8, device=device)

    # Round and convert to int
    coords_int = pred_tracks.round().long()  # (B, T, P, 2)

    for b in range(B):
        for t in range(T):
            pts = coords_int[b, t]  # (P, 2)
            # Filter valid points
            x, y = pts[:, 0], pts[:, 1]
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            pts_valid = pts[valid].cpu().numpy()

            # Create a blank numpy mask for filling
            mask_np = np.zeros((height, width), dtype=np.uint8)

            if not return_mask:
                # If we don't want to return the mask, just set the mask to 1 for valid points
                mask_np[pts_valid[:, 1], pts_valid[:, 0]] = 1

            else:
                if pts_valid.shape[0] >= 3:
                    # Compute convex hull of the points
                    hull = cv2.convexHull(pts_valid.astype(np.int32))  # (H,1,2)
                    # Fill the hull polygon
                    cv2.fillPoly(mask_np, [hull], color=1)
                elif pts_valid.shape[0] > 0:
                    # Fallback: draw small circles for each point
                    for px, py in pts_valid:
                        cv2.circle(mask_np, (int(px), int(py)), radius=1, color=1, thickness=-1)

            # Convert back to torch and place in masks
            masks[b, t] = torch.from_numpy(mask_np).to(device)

    return masks


def get_points_on_a_grid(
        size: int,
        extent: Tuple[float, ...],
        center: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def extend_pointgrid(pointmask: torch.Tensor, grid_size: int) -> torch.Tensor:
    H, W = pointmask.shape
    points = torch.nonzero(pointmask, as_tuple=False).cpu().numpy()  # Shape: (N, 2)
    # Ensure the points are in the correct format for cv2.convexHull
    if points.shape[0] < 3:
        warnings.warn("Not enough points to compute a convex hull. At least 3 points are required.")
        return pointmask

    # Convert points to int32 for cv2.convexHull
    pm_points = points.astype(np.int32)
    pointmask_chull = cv2.convexHull(pm_points.astype(np.int32))
    height, width = pointmask.shape
    # Create a blank binary mask
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    # Fill the convex hull on the binary mask
    cv2.fillPoly(binary_mask, [pointmask_chull], color=255)

    binary_mask = torch.from_numpy(binary_mask).to(pointmask.device).bool()

    new_grid_size = torch.sqrt((torch.sum(pointmask) / torch.sum(binary_mask)) * H * W).item()
    # If new grid size is infinite or NaN, set it to the original grid size
    if not torch.isfinite(torch.Tensor([new_grid_size])) or new_grid_size <= 0:
        new_grid_size = grid_size

    new_grid_size = min(new_grid_size, grid_size * 1.5)

    point_grid = get_points_on_a_grid(
        size=int(new_grid_size),
        extent=(pointmask.shape[0], pointmask.shape[1]),
        device=pointmask.device
    )

    points_from_binary_mask = binary_mask[
        (point_grid[0, :, 1]).round().long().cpu(),
        (point_grid[0, :, 0]).round().long().cpu(),
    ].bool()

    grid_pts = point_grid[:, ~points_from_binary_mask]

    extention_point_grid = torch.zeros_like(pointmask, dtype=torch.uint8)
    extention_point_grid[grid_pts[0, :, 1].round().long(), grid_pts[0, :, 0].round().long()] = 1

    # Combine the original pointmask with the extended point grid
    extended_pointmask = pointmask | extention_point_grid
    return extended_pointmask


def compute_point_mask_iou(pointmask: torch.Tensor, mask: torch.Tensor, grid_size: int) -> float:
    """
    Compute the Intersection between a point mask and a binary mask.

    Args:
        pointmask: torch.Tensor of shape (H, W), dtype uint8, where 1 indicates points.
        mask: torch.Tensor of shape (H, W), dtype uint8, where 1 indicates the mask region.

    Returns:
        IoU value (float) computed as intersection / union.
    """
    # Ensure the masks are boolean
    pointmask = pointmask.bool().to(mask.device)

    extended_pointmask = extend_pointgrid(pointmask, grid_size)

    mask = mask.bool() * extended_pointmask  # Ensure mask is only computed where pointmask is True

    # Compute intersection and union
    intersection = torch.sum(pointmask & mask).item()
    union = torch.sum(pointmask | mask).item()

    # If union is zero, set IoU to 0
    if union == 0:
        return 0.0
    return intersection / union


def compute_point_mask_intersection(pointmask: torch.Tensor, mask: torch.Tensor, grid_size: int) -> float:
    """
    Compute the Intersection between a point mask and a binary mask.

    Args:
        pointmask: torch.Tensor of shape (H, W), dtype uint8, where 1 indicates points.
        mask: torch.Tensor of shape (H, W), dtype uint8, where 1 indicates the mask region.

    Returns:
        IoU value (float) computed as intersection / union.
    """
    # Ensure the masks are boolean
    pointmask = pointmask.bool().to(mask.device)
    mask = mask.bool() * pointmask  # Ensure mask is only computed where pointmask is True

    # Compute intersection and union
    intersection = torch.sum(pointmask & mask).item()
    union = torch.sum(pointmask | mask).item()

    # If union is zero, set IoU to 0
    if union == 0:
        return 0.0
    return intersection / union


def extract_mask_matches(segm_mask, pred_tracks, all_video_masks, frame_id, v_range, grid_size, globalid_lookup,
                         clusterid_lookup,
                         cluster_id, matching_threshold):

    track_masks = pred_tracks_to_binary_masks(pred_tracks, segm_mask.shape[0], segm_mask.shape[1], return_mask=False)

    assert track_masks.shape[1] == all_video_masks.shape[0], \
        f"Track masks shape {track_masks.shape[1]} does not match video masks shape {all_video_masks.shape[0]}"

    vrange_track_masks = track_masks[0,
                         v_range[0]:v_range[1] + 1]  # Get all masks within the visibility range for the track
    vrange_video_masks = all_video_masks[v_range[0]:v_range[1] + 1]  # Get all masks within the visibility range

    matches = []
    all_comparisons = []
    ious = []
    for fid, (tmask, vmasks) in enumerate(zip(vrange_track_masks, vrange_video_masks)):
        frame_object_ids = torch.sort(torch.unique(vmasks[..., 0])[1:])[
            0]  # Frame ids without 0 (background)

        for oid in frame_object_ids:
            # Get the mask for the object ID
            vmask = get_segmentation_mask(vmasks, -1, oid)
            vmask = torch.nn.functional.interpolate(vmask.float(), size=(segm_mask.shape[0], segm_mask.shape[1]),
                                                    mode='nearest').to(torch.uint8)

            # Compute IoU between the track mask and the video mask
            iou = compute_point_mask_intersection(tmask, vmask.squeeze(0).squeeze(0), grid_size)
            ious.append({
                "frame_id": v_range[0] + tmask.shape[0],
                "mask_id": oid.item(),
                "iou": iou
            })


            all_comparisons.append(
                {
                    "frame_id": v_range[0] + fid,
                    "mask_id": oid.item(),
                    "overall_mask_id": get_overall_maskid(globalid_lookup, v_range[0] + fid, oid.item()),
                    "cluster_mask_id": get_cluster_maskid(clusterid_lookup, cluster_id, v_range[0] + fid, oid.item()),
                    "iou": iou
                }
            )

            if iou > matching_threshold:  # Threshold for a match, originally 0.5. Before 0.9
                matches.append({
                    "frame_id": v_range[0] + fid,
                    "mask_id": oid.item(),
                    "overall_mask_id": get_overall_maskid(globalid_lookup, v_range[0] + fid, oid.item()),
                    "cluster_mask_id": get_cluster_maskid(clusterid_lookup, cluster_id, v_range[0] + fid, oid.item()),
                    "iou": iou
                })

    return matches, all_comparisons


def crop_bool_tensor(bool_arr: np.ndarray):
    """
    Crops a 2D boolean numpy array so that only the minimal bounding rectangle
    containing all True (1) values remains.

    Returns:
      cropped_arr: The cropped boolean array.
      offset: A tuple (row_offset, col_offset) indicating the top-left corner
              of the crop in the original array. To map a coordinate (i, j)
              in cropped_arr back to the original array, use:
                  original_i = i + row_offset
                  original_j = j + col_offset
    """
    if bool_arr.ndim != 2:
        raise ValueError("Input must be a 2D boolean array.")

    # Find which rows contain at least one True
    row_has_true = np.any(bool_arr, axis=1)
    # Find which columns contain at least one True
    col_has_true = np.any(bool_arr, axis=0)

    # If there are no True values at all, return an empty crop
    if not row_has_true.any() or not col_has_true.any():
        # You can decide how to handle this case. Here, we'll return an empty array
        # and offsets (0, 0).
        empty_crop = np.zeros((0, 0), dtype=bool)
        return empty_crop, (0, 0)

    # Get the first and last indices along each dimension where True exists
    row_indices = np.where(row_has_true)[0]
    col_indices = np.where(col_has_true)[0]
    row_min, row_max = row_indices[0], row_indices[-1]
    col_min, col_max = col_indices[0], col_indices[-1]

    # Crop the array
    cropped_arr = bool_arr[row_min:row_max + 1, col_min:col_max + 1]

    # The offset tells us how many rows/cols we skipped at the top and left
    offset = (row_min, col_min)
    return cropped_arr, offset


def temporal_correspondance_clustering(matches_data, frameid_maskid_to_overall_maskid_lookup, debug):

    # This function should group together all masks that have been temporally matched for a visibility range
    visibility_and_temporal_data = []

    # Get max overall mask ID for the video
    max_overall_mask_id = max(
        [mask_data["overall_mask_id"] for match in matches_data for mask_data in match["matches"]],
        default=-1
    )

    cluster_ids = sorted(set(m["cluster_id"] for m in matches_data))
    for cid in cluster_ids:
        cluster_matches_data = [mdata for mdata in matches_data if int(mdata["cluster_id"]) == cid]
        cluster_match_matrix = np.zeros((max_overall_mask_id + 1, max_overall_mask_id + 1), dtype=np.float32)
        for i_m, match in enumerate(cluster_matches_data):
            ref_overall_mask_id = match["overall_mask_id"]
            for m in match["matches"]:
                overall_mask_id = m["overall_mask_id"]

                if ref_overall_mask_id >= cluster_match_matrix.shape[0] or overall_mask_id >= cluster_match_matrix.shape[1]:
                    warnings.warn("Overall mask ID exceeds matrix dimensions. Skipping this match.")
                    continue
                cluster_match_matrix[ref_overall_mask_id, overall_mask_id] = 1

        # Cut the cluster match matrix horizontally and vertically
        # Get the first 1 value row-wise and column-wise
        cluster_match_matrix, (row_offset, col_offset) = crop_bool_tensor(cluster_match_matrix)

        print("Shape", cluster_match_matrix.shape) if debug else None

        if cluster_match_matrix.shape[1] > 50:
            _eps = 0.05
            min_samples = 5
        elif cluster_match_matrix.shape[1] < 10:
            _eps = 0.1
            min_samples = 3
        else:
            _eps = 0.1
            min_samples = 5

        print("Clustering with eps:", _eps) if debug else None
        if cluster_match_matrix.shape[0] == 0 or cluster_match_matrix.shape[1] == 0:
            return -1, -1

        clustering = DBSCAN(eps=_eps, min_samples=min_samples, metric="hamming").fit(cluster_match_matrix)
        labels = torch.from_numpy(clustering.labels_)

        # Assign label -1 if sum of row is 0
        for i in range(cluster_match_matrix.shape[0]):
            if cluster_match_matrix[i].sum() == 0:
                labels[i] = -1

        print(f"Clustering labels for cluster {cid}: {labels}") if debug else None

        # Simplicity Score
        visibility_to_temporal_factor = len(set(labels[labels != -1].tolist()))  # Number of unique labels excluding -1
        print(f"Visibility-to-Temporal Factor for cluster {cid}: {visibility_to_temporal_factor}") if debug else None

        overall_mask_ids_per_label = {}
        for i, label in enumerate(labels):
            if label.item() == -1:
                continue
            if label.item() not in overall_mask_ids_per_label:
                overall_mask_ids_per_label[label.item()] = []
            overall_mask_ids_per_label[label.item()].append(
                get_frameid_maskid_from_overall_maskid(frameid_maskid_to_overall_maskid_lookup, i + row_offset))

        print(f"Overall mask IDs per label for cluster {cid}: {overall_mask_ids_per_label}") if debug else None

        visibility_and_temporal_data.append({
            "cluster_id": cid,
            "visibility_to_temporal_factor": visibility_to_temporal_factor,
            "overall_mask_ids_per_label": overall_mask_ids_per_label
        })

    return cluster_ids, visibility_and_temporal_data


def calculate_cluster_coverage(cluster_masks, mask_groupings):
    # This function should calculate the coverage of the cluster masks within the matched masks,
    # e.g. how many of the visibility cluster masks are successfully matched by temporal correspondence
    overall_matched_count = 0
    overall_total_count = 0
    cl_coverages = []
    for c_masks, c_mask_grouping in zip(cluster_masks, mask_groupings):
        if len(c_masks) == 0:
            print("No cluster masks found for this cluster.")
            continue

        all_c_masks = [(int(m["frame_id"]), int(m["mask_id"])) for m in c_masks]
        matched_c_masks = [m for cm in c_mask_grouping["overall_mask_ids_per_label"].values() for m in cm]

        # See how many tuples match
        matched_count = sum(1 for m in matched_c_masks if m in all_c_masks)
        total_count = len(all_c_masks)

        coverage = matched_count / total_count if total_count > 0 else 0
        print(
            f"Cluster coverage: {coverage:.2%} ({matched_count}/{total_count}) for cluster masks {c_masks[0]['vis_cluster_id']}")
        cl_coverages.append(coverage)

        overall_matched_count += matched_count
        overall_total_count += total_count

    overall_coverage = overall_matched_count / overall_total_count if overall_total_count > 0 else 0
    print(f"Overall coverage {overall_coverage:.2%} ({overall_matched_count}/{overall_total_count})")

    return overall_coverage, cl_coverages


def gather_and_save_one2x_data(matches_data, mask_groupings, visibility_group_mask_path):
    # One2x data for visibility clusters
    unique_cluster_ids = sorted(set(m["cluster_id"] for m in matches_data))
    one2x_per_cluster = {}
    for cid in unique_cluster_ids:
        one2x_per_cluster[f"cluster_{cid}"] = [m["one2x"] for m in matches_data if m["cluster_id"] == cid]


    # One2x data for groupings
    video_data = {}
    for c_maskgroup_data in mask_groupings:
        cid = c_maskgroup_data["cluster_id"]
        vis_cluster_path = os.path.join(visibility_group_mask_path, f"cluster_{cid}")
        c_maskgroup = c_maskgroup_data["overall_mask_ids_per_label"]

        gathered_data = {}
        for m_group, fid_mid_list in c_maskgroup.items():
            gathered_data[f"group_{m_group}"] = []
            for (fid, mid) in fid_mid_list:
                # Get the one2x entry for the current frame_id and mask_id
                one2x_entry = next((m["one2x"] for m in matches_data if m["frame_id"] == fid and m["mask_id"] == mid), None)
                if one2x_entry is not None:
                    gathered_data[f"group_{m_group}"].append(one2x_entry)

        output_data = {}
        output_data["avg_one2x_cluster"] = np.mean(one2x_per_cluster.get(f"cluster_{cid}", []))
        for group_name, one2x_entries in gathered_data.items():
            noisy = False
            avg_one2x = np.sum(one2x_entries) / len(one2x_entries) if one2x_entries else 0
            if avg_one2x > 0.5:
                noisy = True
            output_data[group_name] = {
                "avg_one2x": avg_one2x,
                "one2x_counts": len(one2x_entries),
                "noisy": noisy
            }

        # Save the gathered data to a JSON file
        file_path = os.path.join(vis_cluster_path, f"one2x_data_cluster{cid}.json")
        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=4)

        video_data[f"cluster_{cid}"] = output_data
        video_data[f"cluster_{cid}"]["avg_one2x_cluster"] = np.mean(one2x_per_cluster.get(f"cluster_{cid}", []))

    with open(os.path.join(visibility_group_mask_path, "video_one2x_data.json"), "w") as f:
        json.dump(video_data, f, indent=4)




def temporal_correspondence_match(video_path, mask_path, cluster_mask_path, visibility_maps_output_base, visibility_clusters_output_base, matching_threshold, debug=False):
    print("Loading masks...") if debug else None
    all_video_masks = load_masks(mask_path)

    if all_video_masks is None:
        print("Failed to load masks for temporal correspondence matching...")
        return -1

    frameid_maskid_to_overall_maskid_lookup = contruct_frameid_maskid_lookup(all_video_masks)

    cluster_masks = load_cluster_masks(cluster_mask_path)
    if len(cluster_masks) == 0:
        return -1

    frameid_maskid_to_cluster_maskid_lookup = contruct_frameid_maskid_cluster_lookup(cluster_masks)
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

    visibility_maps_output_dir = os.path.join(visibility_maps_output_base, dataset_name, split)
    visibility_clusters_output_dir = os.path.join(visibility_clusters_output_base, dataset_name,
                                                  split)

    if not os.path.exists(visibility_maps_output_dir):
        os.makedirs(visibility_maps_output_dir)

    video_name = os.path.basename(video_path)
    print("Loading video...") if debug else None
    video = mp4_from_images(video_path)
    print("Video loaded.") if debug else None


    clusters_and_visibility_ranges = load_visibility_data(visibility_clusters_output_dir, video_name)

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

    # Sort clusters_and_visibility_ranges by cluster_id
    clusters_and_visibility_ranges.sort(key=lambda x: int(x["cluster_id"]))

    matches_data = []

    if len(cluster_masks) != len(clusters_and_visibility_ranges):
        warnings.warn(
            f"Cluster masks length {len(cluster_masks)} does not match visibility ranges length {len(clusters_and_visibility_ranges)}")
        return -1

    idx_correction = 0

    for cluster_data in tqdm(clusters_and_visibility_ranges):
        cluster_id = cluster_data["cluster_id"]
        visibility_ranges = cluster_data["ranges"]

        # Detect if visibility ranges is an empty sequence
        if len(visibility_ranges) == 0:
            print(f"No visibility ranges for cluster {cluster_id}. Skipping.")
            print(len(cluster_masks), len(clusters_and_visibility_ranges)) if debug else None
            if len(cluster_masks) < len(clusters_and_visibility_ranges):
                idx_correction+=1
            continue

        merge_vranges = True
        if merge_vranges:
            vrange_min = min([v[0] for v in visibility_ranges])
            vrange_max = max([v[1] for v in visibility_ranges])

            visibility_ranges = [(vrange_min, vrange_max)]  # For now, we only use the first range

        print(f"Processing cluster {cluster_id} with ranges {visibility_ranges}. Merge Vranges is {merge_vranges}.") if debug else None

        cluster_masks_and_data = cluster_masks[cluster_id-idx_correction] #if cluster_id >= len(cluster_masks) else cluster_masks[cluster_id]
        print(cluster_masks_and_data) if debug else None

        if len(cluster_masks_and_data) == 0:
            print(f"No masks found for cluster {cluster_id}. Skipping.")
            continue

        if cluster_masks_and_data[0]["vis_cluster_id"] != cluster_id:
            print(f"Cluster ID mismatch: {cluster_masks_and_data[0]['vis_cluster_id']} != {cluster_id}")
            return -1

        # Iterate over visibility ranges
        for v_range in visibility_ranges:
            # This contains the masks with metadata, i.e. frame_id, mask_id,

            visible_cluster_masks_and_data = get_masks_for_vrange(cluster_masks_and_data, v_range)

            for mask_data in tqdm(sorted(visible_cluster_masks_and_data, key=lambda x: int(x["frame_id"])),
                                  desc=f"Processing Masks for Cl. {cluster_id} For Temporal Correspondance Matching"):
                segm_mask, frame_id, mask_id = mask_data["mask"], mask_data["frame_id"], mask_data["mask_id"]
                print(f"Processing mask {mask_id} for frame {frame_id}.") if debug else None

                backward_tracking = frame_id > v_range[0]
                forward_tracking = frame_id < v_range[1] - 1

                upper_grid_size = min(int(np.sum(segm_mask / 255) // 800), 50)
                grid_size = max(upper_grid_size, 25)  # Example heuristic, adjust as needed
                print(f"Grid size for mask {mask_id} in frame {frame_id}: {grid_size}") if debug else None

                pred_tracks, _pred_visibility = model(video, grid_size=grid_size, grid_query_frame=frame_id,
                                                      segm_mask=torch.from_numpy(segm_mask).unsqueeze(0).unsqueeze(
                                                          0), backward_tracking=backward_tracking)

                matches, all_comparisons = extract_mask_matches(segm_mask, pred_tracks, all_video_masks, frame_id,
                                                                v_range,
                                                                grid_size,
                                                                frameid_maskid_to_overall_maskid_lookup,
                                                                frameid_maskid_to_cluster_maskid_lookup, cluster_id-idx_correction, matching_threshold)

                # Check if this mask has been matched to multiple masks from one frame
                one_to_n = {}
                one_to_n_ious = {}
                for comp in all_comparisons:
                    if comp["iou"] > 0.25:
                        if comp["frame_id"] not in one_to_n:
                            one_to_n[comp["frame_id"]] = []
                            one_to_n_ious[comp["frame_id"]] = []
                        one_to_n[comp["frame_id"]].append(comp["mask_id"])
                        one_to_n_ious[comp["frame_id"]].append(comp["iou"])

                warnings_counter = 0
                for frame_i, mask_is in one_to_n.items():
                    if len(mask_is) > 1:
                        warnings_counter += 1


                if warnings_counter >= 5:
                    warnings.warn(
                        f"Too many warnings for multiple matches in frame {frame_id} for mask {mask_id}, check the data quality!") if debug else None

                matches_data.append(
                    {
                        "cluster_id": cluster_id,
                        "frame_id": frame_id,
                        "mask_id": mask_id,
                        "overall_mask_id": get_overall_maskid(
                            frameid_maskid_to_overall_maskid_lookup, frame_id, mask_id),
                        "cluster_mask_id": get_cluster_maskid(
                            frameid_maskid_to_cluster_maskid_lookup, cluster_id-idx_correction, frame_id, mask_id),
                        "one2x": 1 if warnings_counter >= 5 else 0,
                        "matches": matches
                    }
                )
                print(f"Matches for mask {mask_id} in frame {frame_id}: {matches}") if debug else None

    # Okay now we have all the matches and will extract the temporal correspondence cluster
    cluster_ids, mask_groupings = temporal_correspondance_clustering(matches_data,frameid_maskid_to_overall_maskid_lookup, debug)
    if cluster_ids == -1 or mask_groupings == -1:
        return -1

    save_temporal_group_masks(mask_groupings, cluster_masks, cluster_mask_path, idx_correction)

    # How many of the matched masks from the video masks are also part of the cluster masks?
    # If all visibility cluster masks are also part of the matched cluster masks, we have a perfect match
    # We can derive a confidence score from this, e.g. 0.8 if 80% of the masks are matched
    # This can be used to filter for videos with high quality annotations
    video_coverage, cluster_coverages = calculate_cluster_coverage(cluster_masks, mask_groupings)
    visibility_to_temporal_factors = [m["visibility_to_temporal_factor"] for m in mask_groupings]
    save_cluster_coverages(video_coverage, cluster_coverages, visibility_to_temporal_factors, cluster_mask_path)

    # Lastly, see how many % of masks of a group have a one2x flag. Then, save this data to a file
    gather_and_save_one2x_data(matches_data, mask_groupings, cluster_mask_path)


    return 1