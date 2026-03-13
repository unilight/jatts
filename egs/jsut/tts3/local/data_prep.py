#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation script for JSUT.
Here we try to use only basic5000
"""

import argparse
import csv
import os

import jaconv
import librosa
import pyopenjtalk
from jatts.utils.utils import write_csv
from tqdm import tqdm
from collections import defaultdict
import random

TRIM_THRESHOLD_IN_DB = 30
TRIM_FRAME_SIZE = 2048
TRIM_HOP_SIZE = 300

DURAION_MIN = 4.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, required=True, help="database root")
    parser.add_argument(
        "--train_set", type=str, default="train", help="name of train set"
    )
    parser.add_argument("--dev_set", type=str, default="dev", help="name of dev set")
    parser.add_argument("--test_set", type=str, default="test", help="name of test set")
    parser.add_argument(
        "--num_dev", type=int, default=250, help="number of dev set utterances"
    )
    parser.add_argument("--num_test", type=int, default=250, help="number of test set")
    parser.add_argument(
        "--outdir", type=str, required=True, help="output directory path"
    )
    args = parser.parse_args()

    # read all text
    with open(os.path.join(args.db_root, "basic5000", "transcript_utf8.txt"), "r") as f:
        lines = f.read().splitlines()
    texts = {line.split(":")[0]: line.split(":")[1] for line in lines}

    # fieldnames: sample_id,wav_path,original_text
    # format: BASIC5000_0001.wav
    data = defaultdict(list)
    for i in tqdm(range(5000)):
        sample_id = f"BASIC5000_{i+1:04d}"
        wav_path = os.path.join(args.db_root, "basic5000", "wav", sample_id + ".wav")
        original_text = texts[sample_id]
        phns = pyopenjtalk.g2p(original_text)

        # trim silence
        y, sr = librosa.load(wav_path, sr=None)
        _, (start, end) = librosa.effects.trim(
            y,
            top_db=TRIM_THRESHOLD_IN_DB,
            frame_length=TRIM_FRAME_SIZE,
            hop_length=TRIM_HOP_SIZE,
        )

        item = {
            "sample_id": sample_id,
            "spk": "jsut",
            "wav_path": wav_path,
            "original_text": original_text,
            "phonemes": phns,
            "start": start / sr,
            "end": end / sr,
        }

        if i < 5000 - args.num_dev - args.num_test:
            data["train"].append(item)
        elif i >= 5000 - args.num_dev - args.num_test and i < 5000 - args.num_test:
            data["dev"].append(item)
        else:
            data["test"].append(item)

    # add prompt
    for _set in ["train", "dev", "test"]:
        print(f"Adding prompt for {_set} set")
        for i in range(len(data[_set])):
            # Randomly pick one sample from train set
            random_sample = random.choice(data["train"])

            # Ensure the sample_id is not the same
            while random_sample["sample_id"] == item["sample_id"] and DURATION_MIN <= random_sample["end"] - random_sample["start"]:
                random_sample = random.choice(data["train"])

            # Use the "wav_path" as "prompt_wav_path", "start" as "prompt_start", "end" as "prompt_end"
            data[_set][i]["prompt_wav_path"] = random_sample["wav_path"]
            data[_set][i]["prompt_start"] = random_sample["start"]
            data[_set][i]["prompt_end"] = random_sample["end"]

    # write to out
    write_csv(data["train"], os.path.join(args.outdir, args.train_set + ".csv"))
    write_csv(data["dev"], os.path.join(args.outdir, args.dev_set + ".csv"))
    write_csv(data["test"], os.path.join(args.outdir, args.test_set + ".csv"))
