"""Prepare data manifest for pathological video data.

    Slide class: frozen section.
    Sample number: about 700+.
    Classification number: 2.
"""
import csv
import os
import sys

import pandas as pd
import glob

def prepare_frozen_section_manifest():
    """Prepare data manifest for pathological video data."""

    root = "/z3/home/JadeSprings/frozenSection/result/embedding"

    # get valid slides (without labels)
    video_dir_list = glob.glob(os.path.join(root, "*"))
    slide_list = [v.split("/")[-1] for v in video_dir_list]
    group_ids_list = [s[:9] for s in slide_list]

    # create slides and labels mapping
    csv_file = "./data/frozen_section/frozen_section.csv"
    with open(csv_file, "r") as f:
        data = list(csv.DictReader(f))

    t, raw_labels = zip(*[d.values() for d in data])
    raw_slides = [s.split("_")[0] for s in t]
    print(raw_slides[:3])


    label_list = []
    for s in slide_list:
        if s in raw_slides:
            idx = raw_slides.index(s)
            label_list.append(raw_labels[idx])

    print("slide_list: ", slide_list[:5])
    print("group_ids_list: ", group_ids_list[:5])
    print("label_list: ", label_list[:5])

    df = pd.DataFrame({"slide": slide_list, "group_id": group_ids_list, "label": label_list})
    df.to_csv("./data/frozen_section/manifest.csv", index=False)




def test():
    csv_path = "./data/frozen_section/manifest.csv"
    import csv
    with open(csv_path, "r") as f:
        data = list(csv.DictReader(f))
        print(data[:5])
    raw_groups = [d['group_id'] for d in data]
    print(raw_groups[:5])
    import numpy as np
    groups = np.asarray(raw_groups)
    indices = np.asarray(range(len(raw_groups)))

    unique_groups = np.unique(groups[indices])
    print(unique_groups[:5])

if __name__ == "__main__":
    # prepare_frozen_section_manifest()
    test()