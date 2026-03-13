#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Add prompt from a random sample in the training set.
"""

import argparse
import random
from jatts.utils.utils import read_csv, write_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="input csv file")
    parser.add_argument(
        "--prompt_pool_csv",
        type=str,
        required=True,
        help="prompt pool csv file. A random sample of this csn file will be added to each of the sample ",
    )
    parser.add_argument("--out", type=str, required=True, help="output file path")
    args = parser.parse_args()

    # read original csv file
    original_csv, _ = read_csv(args.csv, dict_reader=True)

    # read train csv file
    prompt_pool_csv, _ = read_csv(args.prompt_pool_csv, dict_reader=True)

    data = []
    for item in original_csv:
        # Randomly pick one sample from prompt_pool_csv
        random_sample = random.choice(prompt_pool_csv)

        # Ensure the sample_id is not the same
        while random_sample["sample_id"] == item["sample_id"] or random_sample["spk"] != item["spk"]:
            random_sample = random.choice(prompt_pool_csv)

        # Use the "wav_path" as "prompt_wav_path", "start" as "prompt_start", "end" as "prompt_end"
        item["prompt_wav_path"] = random_sample["wav_path"]
        item["prompt_sample_id"] = random_sample["sample_id"]
        item["prompt_spk"] = random_sample["spk"]
        item["prompt_original_text"] = random_sample["original_text"]
        item["prompt_phonemes"] = random_sample["phonemes"]
        item["prompt_start"] = random_sample["start"]
        item["prompt_end"] = random_sample["end"]

        data.append(item)

    # write to out
    write_csv(data, args.out)
