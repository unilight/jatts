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
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import librosa
import pyopenjtalk
from jatts.utils.utils import write_csv
from tqdm import tqdm

TRIM_THRESHOLD_IN_DB = 40
TRIM_FRAME_SIZE = 4096
TRIM_HOP_SIZE = 600

SETS = ["train_parallel", "train_non_parallel", "dev", "eval"]


def process_sample(args, _set, sample_id, original_text):
    wav_path = os.path.join(args.db_root, "wav", _set, sample_id + ".wav")
    phns = pyopenjtalk.g2p(original_text)

    try:
        y, sr = librosa.load(wav_path, sr=None)
        _, (start, end) = librosa.effects.trim(
            y,
            top_db=TRIM_THRESHOLD_IN_DB,
            frame_length=TRIM_FRAME_SIZE,
            hop_length=TRIM_HOP_SIZE,
        )
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

    item = {
        "sample_id": sample_id,
        "spk": "female",
        "wav_path": wav_path,
        "original_text": original_text,
        "phonemes": phns,
        "start": start / sr,
        "end": end / sr,
        "set": _set,
    }

    return item

def process_sample_wrapper(arg_tuple):
    args, _set, sample_id, original_text = arg_tuple
    return process_sample(args, _set, sample_id, original_text)


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
    data = {"train": [], "dev": [], "test": []}
    with ProcessPoolExecutor() as executor:
        arg_tuples = [(args, _set, sample_id, original_text) for _set in SETS for sample_id, original_text in texts[_set].items()]
        results = list(tqdm(executor.map(process_sample_wrapper, arg_tuples), total=len(arg_tuples)))
    for item in results:
        if item is None:
            continue
        if item["set"].startswith("train"):
            data["train"].append(item)
        elif item["set"] == "dev":
            data["dev"].append(item)
        elif item["set"] == "eval":
            data["test"].append(item)

    for _set in data:
        for i in range(len(data[_set])):
            prompt_index = random.randint(0, len(data["train"]) - 1)
            data[_set][i]["prompt_sample_id"] = data["train"][prompt_index]["sample_id"]
            data[_set][i]["prompt_wav_path"] = data["train"][prompt_index]["wav_path"]
            data[_set][i]["prompt_original_text"] = data["train"][prompt_index][
                "original_text"
            ]
            data[_set][i]["prompt_phonemes"] = data["train"][prompt_index]["phonemes"]
            data[_set][i]["prompt_start"] = data["train"][prompt_index]["start"]
            data[_set][i]["prompt_end"] = data["train"][prompt_index]["end"]

    # write to out
    write_csv(data["train"], os.path.join(args.outdir, args.train_set + ".csv"))
    write_csv(data["dev"], os.path.join(args.outdir, args.dev_set + ".csv"))
    write_csv(data["test"], os.path.join(args.outdir, args.test_set + ".csv"))
