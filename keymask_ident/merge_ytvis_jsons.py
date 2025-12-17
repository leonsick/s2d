#!/usr/bin/env python3
"""
merge_youtube_vis.py

Merge a folder full of single-video YTVIS-style JSON files
into one dataset-level JSON.
"""

import argparse
import copy
import glob
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(src_dir: str, out_file: str, one2x_threshold: float) -> None:
    src_dir = os.path.abspath(src_dir)
    json_paths = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No *.json files found in {src_dir}")

    merged: Dict[str, Any] = {
        "info": "Merged YouTube-VOS style dataset",
        "licenses": {
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "id": 1,
            "name": "Creative Commons Attribution 4.0 License",
        },
        "videos": [],
        "categories": [
            {"supercategory": "object", "id": 1, "name": "fg"},
        ],
        "annotations": [],
    }

    next_video_id = 1
    next_ann_id = 1

    one2x_filter = one2x_threshold > 0
    noisy = 0

    for jp in tqdm(json_paths):
        data = load_json(jp)

        # --- copy global fields once (if present) ---------------------------
        if not merged.get("info") and data.get("info"):
            merged["info"] = data["info"]
        if not merged.get("licenses") and data.get("licenses"):
            merged["licenses"] = data["licenses"]

        # --- video ----------------------------------------------------------
        if not data.get("videos"):
            print(f"⚠️  {jp} has no videos block -- skipped.")
            continue

        video = copy.deepcopy(data["videos"][0])
        old_video_id = video["id"]
        video["id"] = next_video_id
        merged["videos"].append(video)

        # --- annotations ----------------------------------------------------
        for ann in data.get("annotations", []):
            ann_cpy = copy.deepcopy(ann)

            # Filter one2x
            if one2x_filter:
                one2x_instance = ann_cpy.get("one2x", 0.0)
                if one2x_instance > one2x_threshold:
                    noisy += 1
                    continue

            ann_cpy["id"] = next_ann_id
            ann_cpy["video_id"] = next_video_id
            # force category to 1 == "object"
            ann_cpy["category_id"] = 1
            merged["annotations"].append(ann_cpy)
            next_ann_id += 1

        next_video_id += 1

    # -----------------------------------------------------------------------
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"✅  Merged {len(json_paths)} files → {out_file}")
    print(f"   videos:      {len(merged['videos'])}")
    print(f"   annotations: {len(merged['annotations'])}")
    print(f"   one2x noisy annotations removed: {noisy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge single-video YTVIS JSONs into one dataset JSON."
    )
    parser.add_argument("src_dir", help="Folder containing per-video JSON files")
    parser.add_argument("output", help="Destination merged JSON file")
    parser.add_argument("one2x_threshold", type=float, default=-1.0, help="Threshold for one2x score")
    args = parser.parse_args()
    main(args.src_dir, args.output, args.one2x_threshold)
