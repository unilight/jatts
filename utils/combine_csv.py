#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os
import librosa
from tqdm import tqdm
import soundfile as sf

from jatts.utils.utils import read_csv, write_csv, find_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="directory that contains csv files")
    parser.add_argument("--out", type=str, required=True, help="output csv file")
    args = parser.parse_args()

    csv_paths = find_files(args.csv_dir, "*.csv")

    all_data = []
    for csv_path in csv_paths:
        data, _ = read_csv(csv_path, dict_reader=True)
        for line in data:
            all_data.append(line)

    all_data = sorted(all_data, key=lambda x:x["sample_id"])

    write_csv(all_data, args.out)