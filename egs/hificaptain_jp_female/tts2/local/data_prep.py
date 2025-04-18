#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation script for Hi-Fi Captain Japanese female speaker.
Here we merge train_non_parallel and train_parallel
"""

import argparse
import csv
import os
import random

import librosa
import pyopenjtalk
from jatts.utils.utils import write_csv
from tqdm import tqdm

TRIM_THRESHOLD_IN_DB = 40
TRIM_FRAME_SIZE = 4096
TRIM_HOP_SIZE = 600

SETS = ["train_parallel", "train_non_parallel", "dev", "eval"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, required=True, help="database root")
    parser.add_argument(
        "--train_set", type=str, default="train", help="name of train set"
    )
    parser.add_argument("--dev_set", type=str, default="dev", help="name of dev set")
    parser.add_argument("--test_set", type=str, default="test", help="name of test set")
    parser.add_argument(
        "--outdir", type=str, required=True, help="output directory path"
    )
    args = parser.parse_args()

    # read all text
    texts = {}
    for _set in SETS:
        with open(os.path.join(args.db_root, "text", f"{_set}.txt"), "r") as f:
            lines = f.read().splitlines()
        texts[_set] = {line.split(" ")[0]: line.split(" ")[1] for line in lines}

    # fieldnames: sample_id,wav_path,original_text
    # format: BASIC5000_0001.wav
    train_data, dev_data, test_data = [], [], []
    for _set in SETS:
        for sample_id, original_text in tqdm(texts[_set].items()):
            wav_path = os.path.join(args.db_root, "wav", _set, sample_id + ".wav")
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
                "spk": "female",
                "wav_path": wav_path,
                "original_text": original_text,
                "phonemes": phns,
                "start": start / sr,
                "end": end / sr,
            }

            if _set == "eval":
                # For E2-TTS: randomly choose a training sample as prompt
                prompt_index = random.randint(0, len(train_data) - 1)
                item["prompt_sample_id"] = train_data[prompt_index]["sample_id"]
                item["prompt_wav_path"] = train_data[prompt_index]["wav_path"]
                item["prompt_original_text"] = train_data[prompt_index]["original_text"]
                item["prompt_phonemes"] = train_data[prompt_index]["phonemes"]
                item["prompt_start"] = train_data[prompt_index]["start"]
                item["prompt_end"] = train_data[prompt_index]["end"]

            if _set.startswith("train"):
                train_data.append(item)
            elif _set == "dev":
                dev_data.append(item)
            elif _set == "eval":
                test_data.append(item)

    # write to out
    write_csv(train_data, os.path.join(args.outdir, args.train_set + ".csv"))
    write_csv(dev_data, os.path.join(args.outdir, args.dev_set + ".csv"))
    write_csv(test_data, os.path.join(args.outdir, args.test_set + ".csv"))
