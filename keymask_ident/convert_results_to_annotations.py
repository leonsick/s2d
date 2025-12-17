import json
import argparse
import os
import copy

from pycocotools import mask as mask_util
from tqdm import tqdm


def convert_results_to_annotation(annotation_file_path, gt_annotation_path, results_file_path, score_threshold, output_dir, filename):
    """
    Converts a results.json file to an annotation file, calculating
    bounding boxes and areas from segmentation data.

    Args:
        annotation_file_path (str): The file path for the annotation JSON file.
        gt_annotation_path (str): The path to the ground truth annotation file. This is used just to get the correct video metadata.
        results_file_path (str): The file path for the results JSON file.
        output_dir (str): The directory to save the new JSON file.
    """
    # Load the annotation and results files
    with open(annotation_file_path, 'r') as f:
        merged_data = json.load(f)
    with open(results_file_path, 'r') as f:
        results_data = json.load(f)
    # Load the ground truth annotation file to get video metadata
    with open(gt_annotation_path, 'r') as f:
        gt_data = json.load(f)

    # Create a dictionary to store video metadata for easy lookup
    videos_metadata = {video['id']: video for video in gt_data['videos']}
    video_names = {str(video["file_names"][0]).split('/')[0]: video['id'] for video in gt_data['videos']}

    # Create the structure for the new annotation file
    new_annotation_file = {
        "info": gt_data["info"],
        "licenses": gt_data["licenses"],
        "videos": gt_data["videos"],
        "categories": merged_data["categories"],
        "annotations": []
    }

    print(new_annotation_file["videos"][0])
    print(video_names.keys())

    low_scoring = 0

    # Process each prediction in the results file
    for i, prediction in enumerate(tqdm(results_data, desc=f"Converting annotations for {os.path.basename(results_file_path)}")):
        video_id = prediction["video_id"]

        if prediction["score"] < args.score_threshold:
            print(f"Skipping prediction for video {video_id} due to low score: {prediction['score']}")
            low_scoring+=1
            continue

        if video_id in videos_metadata:
            video_info = videos_metadata[video_id]

            # Initialize lists for bboxes and areas
            num_frames = video_info["length"]

            assert num_frames == len(prediction["segmentations"]), \
                f"Number of frames in video {video_id} ({num_frames}) does not match the number of segmentations ({len(prediction['segmentations'])})"

            bboxes = [None] * len(prediction["segmentations"])
            areas = [None] * len(prediction["segmentations"])

            # Calculate bboxes and areas from segmentations
            for frame_idx, rle in enumerate(prediction["segmentations"]):
                if rle is not None:
                    # Create a deep copy for calculations to avoid modifying the original
                    rle_for_calc = copy.deepcopy(rle)
                    # Ensure counts is bytes for mask_util
                    if isinstance(rle_for_calc['counts'], str):
                        rle_for_calc['counts'] = rle_for_calc['counts'].encode('ascii')

                    #print(f"Processing frame {frame_idx + 1}/{num_frames}. VideoID {video_id}, Boxes {len(bboxes)}, predictionsegmentations: {len(prediction['segmentations'])}")

                    bboxes[frame_idx] = mask_util.toBbox(rle_for_calc).tolist()
                    areas[frame_idx] = int(mask_util.area(rle_for_calc))

            # Create a new annotation dictionary
            new_annotation = {
                "video_id": video_id,
                "iscrowd": 0,
                "height": video_info["height"],
                "width": video_info["width"],
                "length": num_frames,
                "segmentations": prediction["segmentations"],
                "bboxes": bboxes,
                "areas": areas,
                "category_id": prediction["category_id"],
                "id": i + 1  # Assign a unique ID
            }
            new_annotation_file["annotations"].append(new_annotation)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the output file path
    output_file_path = os.path.join(output_dir, f"{filename}.json")


    # Write the new annotation data to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(new_annotation_file, f, indent=2)

    print(f"Successfully converted '{results_file_path}' to '{output_file_path}'")
    print(f"Skipped {low_scoring}/{len(results_data)} ({round((low_scoring/len(results_data))*100, 2)}%) low scoring predictions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a results.json file to a COCO-style annotation file.")
    parser.add_argument('--annotation-file', help="Path to the merged.json file.")
    parser.add_argument('--gt-annotation-file', help="Root directory for the dataset, used to locate video files.")
    parser.add_argument('--results-file', help="Path to the results.json file.")
    parser.add_argument('--score-threshold', type=float, default=0.75, help="Score threshold.")
    parser.add_argument('--output-dir', help="Path to save the output annotation file.")
    parser.add_argument('--output-filename', help="File name.")
    args = parser.parse_args()

    convert_results_to_annotation(args.annotation_file, args.gt_annotation_file, args.results_file, args.score_threshold, args.output_dir, args.output_filename)