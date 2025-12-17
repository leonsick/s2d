import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def visualize_distillation_targets(gt_masks_per_video, images, i, h_pad, w_pad, scores_per_image, labels_per_image, score_threshold, video_rand_int, extra):
    # --- Visualization of gt_masks_per_video on input frames (with per-video random id) ---

    debug_root = "/mnt/hdd/leon/outputs/debug"

    # Generate a unique random int per video to avoid overwriting across videos
    if not os.path.exists(debug_root):
        os.makedirs(debug_root)

    while True:
        video_debug_dir = os.path.join(debug_root, f"video_{video_rand_int}_{extra}")
        try:
            os.makedirs(video_debug_dir)
            break
        except FileExistsError:
            # Extremely rare, but try another random id if it already exists
            continue

    # Stable color palette for up to N objects in this video
    num_objs = int(gt_masks_per_video.shape[0])
    rng = random.Random(1234)
    color_palette = [tuple(rng.randint(50, 255) for _ in range(3)) for _ in range(max(1, num_objs))]

    # Denormalization constants on correct device
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.tensor.device).view(3, 1, 1)

    # Get frames for current video index `i`
    frames_tensor = images.tensor
    if frames_tensor.dim() == 5:
        # [B, T, 3, H, W]
        video_frames = frames_tensor[i]  # -> [T, 3, H, W]
    elif frames_tensor.dim() == 4:
        # [B*T, 3, H, W] or [T, 3, H, W]
        T_expected = gt_masks_per_video.shape[1]
        if frames_tensor.shape[0] == T_expected:
            video_frames = frames_tensor  # single video in batch
        else:
            start = i * T_expected
            end = start + T_expected
            video_frames = frames_tensor[start:end]
    else:
        raise RuntimeError(f"Unexpected images.tensor shape: {frames_tensor.shape}")

    # Overlay masks frame-by-frame
    alpha = 0.5
    T = video_frames.shape[0]
    for f_i in range(T):
        h_pad, w_pad = video_frames.shape[2], video_frames.shape[3]
        # Denormalize image back to [0,1], then to uint8
        img_tensor = video_frames[f_i, :, :h_pad, :w_pad].float()
        img_tensor = torch.clamp(img_tensor * std + mean, 0, 1)
        img_np = (img_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        overlay = img_np.copy()

        # If the image doesnt fit the size of the masks, skip this frame
        if img_np.shape[0] != h_pad or img_np.shape[1] != w_pad:
            print(f"Skipping visualization for frame {f_i} due to size mismatch: img_np.shape={img_np.shape}, expected height={h_pad}, width={w_pad}")

        # Masks for this frame: [num_objs, H, W] (bool)
        frame_masks = gt_masks_per_video[:, f_i, :h_pad, :w_pad]

        print("img_tensor.shape:", img_tensor.shape)
        print("frame_masks.shape:", frame_masks.shape)

        if frame_masks.ndim == 3 and frame_masks.shape[0] > 0:
            for obj_idx in range(frame_masks.shape[0]):
                # if the score is too low, skip visualization
                if scores_per_image[obj_idx] < score_threshold:
                    continue

                mask = frame_masks[obj_idx].detach().cpu().numpy().astype(bool)
                if not mask.any():
                    continue
                color = np.array(color_palette[obj_idx % len(color_palette)], dtype=np.uint8)
                overlay[mask] = ((1.0 - alpha) * overlay[mask] + alpha * color).astype(np.uint8)

        # Save with the per-video random id in both folder and filename
        save_path = os.path.join(video_debug_dir, f"video_{video_rand_int}_frame_{f_i:04d}_overlay.png")
        Image.fromarray(overlay).save(save_path)

        # print the scores and labels for this frame
        print(f"Frame {f_i}:")
        for obj_idx in range(len(scores_per_image)):
            score = scores_per_image[obj_idx].item()
            label = labels_per_image[obj_idx].item()
            print(f"  Object {obj_idx}: Score={score:.4f}, Label={label}")


def debug_visualize_matched_masks(
    outputs,
    targets,
    indices,
    out_dir="/mnt/hdd/leon/outputs/debug/mask_loss",
    threshold: float = 0.5,
    max_pairs: int = 24,
):
    """
    Saves side-by-side visualizations for matched student-vs-teacher masks:

        [ student (sigmoid>thr) | pseudo label | XOR error ]

    One image per frame. Works for video shapes BxQxTxHxW.

    Activate by setting:
        criterion.debug_vis_dir = "path/to/folder"
        (optionally) criterion.debug_vis_thresh, criterion.debug_vis_max

    Args:
        outputs: dict with "pred_masks" (B,Q,T,H,W) logits
        targets: list of dicts of length B; each has 'masks' or 'masks_logits'
        indices: list[(src_idx, tgt_idx)] matches for each batch item
        out_dir: folder to save to
        threshold: binarization threshold for sigmoid(prob)
        max_pairs: cap number of matched pairs saved per call
    """
    # ---- Setup & shapes
    pred_masks = outputs["pred_masks"].detach()
    if pred_masks.dim() == 4:
        # If model outputs 4D, treat as single-frame video
        pred_masks = pred_masks.unsqueeze(2)  # B,Q,1,H,W

    B, Q, T, H, W = pred_masks.shape

    # Per-rank folder (safe for DDP)
    try:
        from detectron2.utils.comm import get_rank
        rank = get_rank()
    except Exception:
        rank = 0
    save_root = os.path.join(out_dir, f"rank{rank}")
    os.makedirs(save_root, exist_ok=True)

    # Running counter across calls (so filenames stay unique)
    counter = 0
    saved = 0

    def to_uint8_img(t2d):
        """t2d: 2D float/byte torch tensor in [0,1] or {0,1} -> PIL 'L' image"""
        arr = (t2d.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        return Image.fromarray(arr, mode="L")

    for b, (src_idx, tgt_idx) in enumerate(indices):
        s_list = src_idx.tolist()
        t_list = tgt_idx.tolist()

        for j, (si, ti) in enumerate(zip(s_list, t_list)):
            if saved >= max_pairs:
                break

            # Student logits -> probs -> binarized per frame
            stud_logits_bqt = pred_masks[b, si]  # (T,H,W)
            stud_probs_bqt = stud_logits_bqt.sigmoid()

            # Teacher / pseudo
            tgt_has_masks = ("masks" in targets[b])
            tgt_has_logits = ("masks_logits" in targets[b])

            if not (tgt_has_masks or tgt_has_logits):
                continue

            if tgt_has_masks:
                # Expect (T,H,W) or (H,W); convert to float tensor
                tgt_mask = targets[b]["masks"][ti].detach().float()
                if tgt_mask.dim() == 2:
                    tgt_mask = tgt_mask.unsqueeze(0).expand(T, -1, -1)  # make T
                # If sizes differ, nearest-neighbor resize to (H,W)
                if tgt_mask.shape[-2:] != (H, W):
                    tgt_mask = torch.stack([
                        F.interpolate(
                            tgt_mask[t].unsqueeze(0).unsqueeze(0),
                            size=(H, W), mode="nearest"
                        )[0, 0]
                        for t in range(T)
                    ], dim=0)
                tgt_bin_bqt = (tgt_mask > 0.5).float()
            else:
                # masks_logits: sigmoid + threshold; bilinear to align
                tgt_logits = targets[b]["masks_logits"][ti].detach()
                if tgt_logits.dim() == 2:
                    tgt_logits = tgt_logits.unsqueeze(0).expand(T, -1, -1)
                # resize logits to (H,W) with bilinear
                tgt_prob_bqt = torch.stack([
                    F.interpolate(
                        tgt_logits[t].unsqueeze(0).unsqueeze(0),
                        size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0].sigmoid()
                    for t in range(T)
                ], dim=0)
                tgt_bin_bqt = (tgt_prob_bqt > threshold).float()

            # Save one image per frame
            for t in range(T):
                stud_bin = (stud_probs_bqt[t] > threshold).float()
                tgt_bin  = tgt_bin_bqt[t]

                # XOR error map (1 where they differ)
                diff = (stud_bin != (tgt_bin > 0.5)).float()

                # IoU for filename
                inter = (stud_bin * tgt_bin).sum().item()
                union = (stud_bin + tgt_bin - stud_bin * tgt_bin).sum().item()
                iou = (inter / (union + 1e-6)) if union > 0 else 0.0

                # Build triptych canvas: [student | teacher | error]
                left  = to_uint8_img(stud_bin)
                mid   = to_uint8_img(tgt_bin)
                right = to_uint8_img(diff)

                canvas = Image.new("L", (W * 3 + 20, H), 0)  # black background
                canvas.paste(left,  (0, 0))
                canvas.paste(mid,   (W + 10, 0))
                canvas.paste(right, (2 * W + 20, 0))

                fname = os.path.join(
                    save_root,
                    f"step{counter:06d}_b{b}_pair{j}_q{si:03d}_tgt{ti:03d}_f{t:02d}_iou{iou:.3f}.png"
                )
                canvas.save(fname)
                saved += 1

            counter += 1

        if saved >= max_pairs:
            break

    _debug_vis_counter = counter
