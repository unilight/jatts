#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Add text (this is a post-hoc code).
"""

import argparse
import csv
import os
import pyopenjtalk
from jatts.utils.utils import read_csv, write_csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# Global shared variable
shared_texts = {}

def init_worker(texts):
    global shared_texts
    shared_texts = texts

def process_sample(item):
    sample_id = item["sample_id"]

    original_text = shared_texts[sample_id]
    phonemes = pyopenjtalk.g2p(original_text)

    item_copy = {k: v for k, v in item.items()}
    item_copy["original_text"] = original_text
    item_copy["phonemes"] = phonemes
    return item_copy

def process_sample_wrapper(arg_tuple):
    texts, item = arg_tuple
    return process_sample(texts, item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv", type=str, required=True, help="in csv file"
    )
    parser.add_argument("--db_root", type=str, required=True, help="database root")
    parser.add_argument("--out", type=str, required=True, help="output file path")
    args = parser.parse_args()

    print("Read all.csv...")
    all_data, _ = read_csv(os.path.join(args.db_root, "all.csv"), dict_reader=True)
    texts = {item["_id"]: item["text"] for item in all_data}

    # read original csv file
    original_csv, _ = read_csv(args.in_csv, dict_reader=True)

    # put text into csv
    data = []
    with ProcessPoolExecutor(initializer=init_worker, initargs=(texts,)) as executor:
        results = list(tqdm(executor.map(process_sample, original_csv), total=len(original_csv)))

    for item in results:
        if item is None:
            continue
        data.append(item)

    # write to out
    write_csv(data, args.out)
