#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import csv
import os

import jaconv
import pyopenjtalk
import yaml
from tqdm import tqdm

from jatts.utils.utils import read_csv, write_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_csv", type=str, required=True, help="original wavscp file"
    )
    parser.add_argument("--out", type=str, required=True, help="out wavscp file")

    args = parser.parse_args()

    # read csv
    original_csv, _ = read_csv(args.original_csv, dict_reader=True)

    # loop
    data = []
    for item in tqdm(original_csv):

        new_item = {k: v for k, v in item.items()}

        g2p_result = pyopenjtalk.g2p(item["original_text"], kana=True)
        hira = jaconv.kata2hira(g2p_result)
        julius_format = jaconv.hiragana2julius(hira)
        normalized_g2p_result = julius_format.replace("。", "").replace("、", " sp ")
        new_item["phonemes"] = normalized_g2p_result

        data.append(new_item)

    write_csv(data, args.out)
