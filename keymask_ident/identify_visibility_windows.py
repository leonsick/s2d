import json
import os

import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def json_to_tensor(json_file_or_data):
    """
    Convert a JSON file to a tensor.
    """

    if isinstance(json_file_or_data, str):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = json_file_or_data

    video_data = data['video_data']

    # Assuming video_data is a list of dictionaries
    tensor_data = []
    for per_frame_data in video_data:
        frame_id = per_frame_data['frame_id']
        frame_data = per_frame_data['data']

        for object_data in frame_data:
            object_id = object_data['object_id']
            visibility = object_data['visibility']
            # Assuming visibility is a list of floats
            tensor_data.append(torch.tensor(visibility))

    # Convert to tensor
    tensor_data = torch.stack(tensor_data)
    return tensor_data

def json_to_lookup_dict(json_file_or_data):
    if isinstance(json_file_or_data, str):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = json_file_or_data

    video_data = data['video_data']

    # Assuming video_data is a list of dictionaries
    lookup_dictionary = []
    for per_frame_data in video_data:
        frame_id = per_frame_data['frame_id']
        frame_data = per_frame_data['data']

        for object_data in frame_data:
            object_id = object_data['object_id']
            visibility = object_data['visibility']
            # Assuming visibility is a list of floats
            lookup_dictionary.append({
                "frame_id": frame_id,
                "object_id": object_id
            })

    return lookup_dictionary


def get_visible_ranges(maj_vote):
    """
    Given a 1D array/ tensor of 0s and 1s, return a list of (start, end)
    index tuples (inclusive) for which the value is 1.
    """
    # convert to a plain Python list of bools
    vis = maj_vote.cpu().numpy().astype(bool).tolist()
    ranges = []
    start = None

    for i, v in enumerate(vis):
        if v and start is None:
            # entering a visible run
            start = i
        elif not v and start is not None:
            # leaving a visible run
            ranges.append((start, i - 1))
            start = None

    # if we ended in a visible run, close it out
    if start is not None:
        ranges.append((start, len(vis) - 1))

    return ranges

def get_highly_visible_rows(cluster_vis, runs, threshold=0.8):
    """
    For each (start,end) in runs, find which rows in cluster_vis
    have visibility fraction > threshold over those frames.
    Returns a dict mapping (start,end) -> list of row‐indices.
    """
    out = {}
    for (start, end) in runs:
        length = end - start + 1
        # sum along each row over the run
        counts = cluster_vis[:, start:end+1].sum(dim=1)
        frac   = counts / length
        # select local row indices
        rows   = (frac > threshold).nonzero(as_tuple=True)[0].tolist()
        out[(start, end)] = rows
    return out


def get_visibility_windows_for_video(video_data, dataset_name, split, video_name, cluster_output_dir, visibility_threshold, debug=False):
    tensor = json_to_tensor(video_data)
    row_to_frameid_maskid = json_to_lookup_dict(video_data)
    #print(tensor.shape)

    # DBSCAN clustering
    clustering = DBSCAN(eps=0.2, min_samples=5, metric="hamming").fit(tensor.numpy() > visibility_threshold) # visibility_threshold 0.3
    labels = torch.from_numpy(clustering.labels_)
    #print(labels)

    # 1. Binarize visibility
    vis_all = (tensor > visibility_threshold).float()  # shape (N, T)

    # 2. Mask out the noise label (-1)
    mask = labels != -1  # shape (N,)
    vis = vis_all[mask]  # shape (N_valid, T)
    labs = labels[mask]  # shape (N_valid,)

    row_to_frameid_maskid = [row_to_frameid_maskid[i] for i in range(len(row_to_frameid_maskid)) if mask[i]]

    # 3. Find all cluster IDs (excluding -1)
    clusters = torch.unique(labs)  # e.g. tensor([0,1,2,3])
    # (since we already filtered -1, no need to re-exclude)

    cluster_data = []

    for l in clusters:
        idxs = (labs == l).nonzero(as_tuple=True)[0]  # (n_i,)
        cluster_vis = vis[idxs]  # (n_i, T)

        # majority-vote per frame:
        # sum up the 0/1’s along dim=0 → shape (T,)
        counts = cluster_vis.sum(dim=0)
        n_i = cluster_vis.size(0)

        # define “majority” as > 50%; use >= if you want ties to go visible
        maj_vote = (counts > (n_i / 2)).float()  # (T,)

        visualize = False
        if visualize:
            # now you can plot it, or store it, etc.
            plt.figure()
            # plot all tracks lightly
            for row in cluster_vis:
                plt.plot(row.cpu().numpy(), alpha=0.1)
            # overplot the majority-vote curve
            plt.plot(maj_vote.cpu().numpy(), color="k", linewidth=2, label="majority")
            plt.title(f"Cluster {int(l)} (n={n_i})")
            plt.xlabel("Frame")
            plt.ylabel("Visible (1) / Occluded (0)")
            plt.ylim(-0.1, 1.1)
            plt.legend()
            plt.show()

        ranges = get_visible_ranges(maj_vote)

        print(f"Visible ranges for cluster {int(l)}: {ranges}") if debug else None

        visibility_winner_masks = get_highly_visible_rows(cluster_vis, ranges, threshold=0.3)

        print(f"Visibility winner masks for cluster {int(l)}: {visibility_winner_masks}") if debug else None

        all_candidates_data = []
        all_visible_masks = []
        for start_end, rows in visibility_winner_masks.items():
            print(f"Start-End: {start_end}, Rows: {rows}") if debug else None
            candidates = []
            for row in rows:
                row = idxs[row]

                all_visible_masks.append(
                    {
                        "frame_id": row_to_frameid_maskid[row]['frame_id'],
                        "mask_id": row_to_frameid_maskid[row]['object_id']
                    }
                )

                if row_to_frameid_maskid[row]['frame_id'] <= start_end[1] and row_to_frameid_maskid[row]['frame_id'] >= \
                        start_end[0]:
                    # print(f"Row {row} corresponds to frame ID {row_to_frameid_maskid[row]['frame_id']} and mask ID {row_to_frameid_maskid[row]['object_id']}")
                    candidates.append(
                        {
                            "start_frame": start_end[0],
                            "end_frame": start_end[1],
                            "frame_id": row_to_frameid_maskid[row]['frame_id'],
                            "mask_id": row_to_frameid_maskid[row]['object_id']
                        }
                    )

            all_candidates_data.append(
                {
                    "range": start_end,
                    "candidates": candidates

                }
            )

        cluster_data.append(
            {
                "cluster_id": int(l),
                "cluster_size": n_i,
                "ranges": ranges,
                "all_candidates": all_candidates_data,
                "all_visible_masks": all_visible_masks
            }
        )


    video_data = {
        "video_name": video_name,
        "clusters": cluster_data
    }

    # Save the video data to a JSON file
    output_json_file = f"{cluster_output_dir}/{dataset_name}/{split}/{video_name}.json"
    if not os.path.exists(output_json_file):
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)

    with open(output_json_file, 'w') as f:
        json.dump(video_data, f, indent=4)

    print("Saved visibility clusters for video:", video_name, "to", output_json_file)

    return video_data




