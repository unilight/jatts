#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os
import librosa
from tqdm import tqdm
import soundfile as sf

from jatts.utils.utils import read_csv, write_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="csv file")
    parser.add_argument("--n_splits", type=int, required=True, help="number of splits")
    parser.add_argument("--outdir", type=str, required=True, help="output dir")
    args = parser.parse_args()

    data, _ = read_csv(args.csv, dict_reader=True)

    k, m = divmod(len(data), args.n_splits)  # k is the size of each chunk, m is the remainder
    
    for i in range(args.n_splits):
        write_csv(data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)],
                  os.path.join(args.outdir, os.path.basename(args.csv).replace(".csv", f".{i+1}.csv"))
                )