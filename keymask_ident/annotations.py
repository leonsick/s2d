import json, os
from PIL import Image
import numpy as np
from pycocotools import mask as mask_util
import re


def write_annotation_for_video(video_path, cluster_masks_path, annotation_output_path, visibility_data):
    """
    Write the visibility annotations for a video in a JSON file.
    """

    # Get the name of the video folder
    video_name = os.path.basename(video_path)

    # Create a list of all filenames with .jpg, .png, or .jpeg extensions in the video folder
    video_files = sorted([
        f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    if not video_files:
        print(f"No image files found in {video_path}")
        return

    with Image.open(os.path.join(video_path, video_files[0])) as img:
        width, height = img.size

    video_data = {
        'license': 1, 'coco_url': '', 'height': height, 'width': width, 'length': len(video_files),
        'date_captured': '2019-04-11 00:55:41.903902',
        'file_names': [os.path.join(video_name, f) for f in video_files], 'flickr_url': '', 'id': 1
    }

    annotations = []
    annotation_id = 1

    cluster_dirs = sorted([
        d for d in os.listdir(cluster_masks_path)
        if os.path.isdir(os.path.join(cluster_masks_path, d)) and
           d.startswith('cluster_') and
           len([f for f in os.listdir(os.path.join(cluster_masks_path, d)) if f.endswith('.png')]) > 0  # Added check for non-empty directory
    ])

    with open(os.path.join(cluster_masks_path, 'video_one2x_data.json'), 'r') as f:
        one2x_data = json.load(f)

    print(f"All cluster directories found {cluster_dirs}")
    print("one2x_data keys:", one2x_data.keys())

    for cluster_name in cluster_dirs:
        cluster_dir_path = os.path.join(cluster_masks_path, cluster_name)
        group_dirs = sorted([d for d in os.listdir(cluster_dir_path) if os.path.isdir(os.path.join(cluster_dir_path, d)) and d.startswith('group_')])

        # Get visibility range
        try:
            # Get the cluster number
            c_id = int(cluster_name.replace('cluster_', ''))
            cluster_vis_data = next((c for c in visibility_data["clusters"] if c["cluster_id"] == c_id), None)
            cluster_visibility_ranges = cluster_vis_data["ranges"]
        except KeyError:
            print("Could not find visibility data.")
            cluster_visibility_ranges = [(-1, -1)]

        # Get one2x data
        try:
            cluster_one2x_data = one2x_data[cluster_name]
        except KeyError:
            print(f"Could not find one2x data for {cluster_name}.")
            continue

        for group_name in group_dirs:
            group_dir_path = os.path.join(cluster_dir_path, group_name)

            group_one2x_data = round(float(cluster_one2x_data[group_name]["avg_one2x"]), 2)

            num_frames = len(video_files)
            segmentations = [None] * num_frames
            bboxes = [None] * num_frames
            areas = [None] * num_frames

            mask_files = [f for f in os.listdir(group_dir_path) if f.endswith('.png')]

            for mask_file in mask_files:
                # Extract frame index from filename like 'frame5_mask1.png'
                match = re.search(r'frame(\d+)', mask_file)
                if not match:
                    continue

                frame_idx = int(match.group(1))

                if frame_idx >= num_frames:
                    continue

                # Read mask and convert to RLE
                mask_image = Image.open(os.path.join(group_dir_path, mask_file)).convert('L')
                binary_mask = np.array(mask_image) > 0

                # The RLE object is a dictionary with 'size' and 'counts'.
                # The 'counts' are bytes and need to be decoded for JSON serialization.
                rle = mask_util.encode(np.array(binary_mask[..., None], order="F", dtype="uint8"))[0]
                rle['counts'] = rle['counts'].decode('ascii')

                segmentations[frame_idx] = rle

                areas[frame_idx] = int(mask_util.area(rle))
                bboxes[frame_idx] = mask_util.toBbox(rle).tolist()

            # Create annotation object for this group (spatio-temporal instance)
            annotation_obj = {
                'video_id': video_data['id'],
                'iscrowd': 0,
                'height': height,
                'width': width,
                'length': num_frames,
                'segmentations': segmentations,
                'bboxes': bboxes,
                'areas': areas,
                'category_id': 1, # Assuming a single category for now
                'id': annotation_id,
                'one2x': group_one2x_data,
                'visibility_ranges': cluster_visibility_ranges,
            }
            annotations.append(annotation_obj)
            annotation_id += 1

    # Final JSON structure
    annotation_data = {
        'videos': [video_data],
        'annotations': annotations,
        'categories': [{'supercategory': 'object', 'id': 1, 'name': 'fg'}]
    }

    if not os.path.exists(annotation_output_path):
        os.makedirs(annotation_output_path)

    annotation_output_path = os.path.join(annotation_output_path, f'{video_name}.json')

    # Write to output file
    with open(annotation_output_path, 'w') as f:
        json.dump(annotation_data, f)