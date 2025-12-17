import os
from tqdm import tqdm

import crw_utils
from cotracker_occlusions import extract_object_visibility_data
from identify_visibility_windows import get_visibility_windows_for_video
from keymask_utils import save_segmentation_masks
from cotracker_matching import temporal_correspondence_match
from annotations import write_annotation_for_video

if __name__ == "__main__":
    args = crw_utils.keymask_args()

    video_base_path = args.video_base_path
    video_paths = [os.path.join(video_base_path, video_name) for video_name in sorted(os.listdir(video_base_path))]

    video_paths = [path for path in video_paths if os.path.isdir(path)]

    # Split per job
    if args.videos_per_job > 0:
        start_idx = args.job_id * args.videos_per_job if args.job_id > 0 else 0
        end_idx = start_idx + args.videos_per_job
        video_paths = video_paths[start_idx:end_idx]

    #video_paths = video_paths[1:]  # For testing purposes, only take the first video

    mask_base_path = args.mask_base_path
    masks_paths = [os.path.join(mask_base_path, video_name) for video_name in sorted(os.listdir(video_base_path))]

    # Make sure mask_paths only contains directories
    masks_paths = [path for path in masks_paths if os.path.isdir(path)]

    # Split per job
    if args.videos_per_job > 0:
        masks_paths = masks_paths[start_idx:end_idx]

    #masks_paths = masks_paths[1:]  # For testing purposes, only take the first video

    if video_base_path.__contains__("DAVIS"):
        dataset_name = "DAVIS"
        split = "all"
    elif video_base_path.__contains__("ytvis2021"):
        dataset_name = "ytvis2021"
        if video_base_path.__contains__("train"):
            split = "train"
        else:
            split = "valid"
    elif video_base_path.__contains__("ytvis2019"):
        dataset_name = "ytvis2019"
        if video_base_path.__contains__("train"):
            split = "train"
        else:
            split = "valid"
    elif video_base_path.__contains__("ovis"):
        dataset_name = "ovis"
        if video_base_path.__contains__("train"):
            split = "train"
        else:
            split = "valid"
    elif video_base_path.__contains__("VIPSeg"):
        dataset_name = "VIPSeg"
        split = "imgs"
    elif video_base_path.__contains__("MOSE"):
        dataset_name = "MOSE"
        if video_base_path.__contains__("train"):
            split = "train"
        else:
            split = "valid"
    elif video_base_path.__contains__("sa-v"):
        dataset_name = "SA-V"
        split = "train"
    else:
        raise ValueError("Unknown dataset name. Please specify the dataset name in the video base path.")

    failed_annotation_extractions = 0

    print("Let's go!")
    print(len(video_paths))
    print(len(masks_paths))

    for video_path, masks_path in tqdm(zip(video_paths, masks_paths), total=len(video_paths), desc="Processing videos"):
        video_name = os.path.basename(video_path)

        print("Processing video:", video_name)

        if os.path.exists(os.path.join(args.annotation_output_path, f"{video_name}.json")):
            print(f"Annotation for video {video_name} already exists. Skipping.")
            continue

        try:
            visibility_data = extract_object_visibility_data(video_path, masks_path, args.video_output_dir, args.visibility_maps_output_base, args.debug)
        except Exception as e:
            print(f"Error during visibility data extraction for video {video_name}: {e}")
            failed_annotation_extractions += 1
            continue

        if visibility_data is None:
            failed_annotation_extractions+=1
            continue

        try:
            visibility = get_visibility_windows_for_video(visibility_data, dataset_name, split, video_name, args.visibility_clusters_output_base, args.visibility_threshold, args.debug)
        except Exception as e:
            print(f"Error during visibility window identification for video {video_name}: {e}")
            failed_annotation_extractions += 1
            continue

        try:
            imgs, imgs_orig, lbls, meta = crw_utils.load_frames_and_masks(video_path, masks_path, visibility, dataset_name)
        except Exception as e:
            print(f"Error loading frames and masks for video {video_name}: {e}")
            failed_annotation_extractions += 1
            continue

        if imgs is None:
            print("Image or Mask Loading Error has occurred. Skipping video as to not crash the entire process.")
            failed_annotation_extractions += 1
            continue

        try:
            cluster_masks_path = save_segmentation_masks(imgs, imgs_orig, lbls, meta, args.save_path, args.debug)
        except Exception as e:
            print(f"Error during segmentation mask saving for video {video_name}: {e}")
            failed_annotation_extractions += 1
            continue

        try:
            status = temporal_correspondence_match(video_path, masks_path, cluster_masks_path, args.visibility_maps_output_base, args.visibility_clusters_output_base, args.matching_threshold, args.debug)
        except Exception as e:
            print(f"Error during temporal correspondence matching for video {video_name}: {e}")
            failed_annotation_extractions += 1
            continue

        if status > 0:
            print("Saving annotations for video:", video_name)
            write_annotation_for_video(video_path, cluster_masks_path, args.annotation_output_path, visibility)
        else:
            print("No valid annotations found for video:", video_name)
            failed_annotation_extractions += 1


    print(f"Final Report: Successful annotations ->{(len(video_paths)-failed_annotation_extractions)}/{len(video_paths)}; Failed annotations ->{failed_annotation_extractions}/{len(video_paths)}. ")

