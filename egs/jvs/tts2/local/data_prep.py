#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation script for JVS.
"""

import argparse
import csv
import os
import librosa
import pyopenjtalk
from jatts.utils.utils import read_csv, write_csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


SET_MAPPING = {"parallel": "parallel100", "nonparallel": "nonpara30"}

TRIM_THRESHOLD_IN_DB = 30
TRIM_FRAME_SIZE = 2048
TRIM_HOP_SIZE = 300

def process_sample(args, texts, item):
    # sample_id looks like jvs001_parallel_VOICEACTRESS100_001
    sample_id = item["sample_id"]
    spk, _set, _id = sample_id.split("_", 2)

    new_data = {k: v for k, v in item.items()}

    # fill in text
    original_text = texts[spk][_id]
    phonemes = pyopenjtalk.g2p(original_text)
    new_data["original_text"] = original_text
    new_data["phonemes"] = phonemes

    # override wav_path to absolute path
    new_data["wav_path"] = os.path.join(args.db_root, item["wav_path"])
    if "ref_wav_path" in item:
        new_data["ref_wav_path"] = os.path.join(args.db_root, item["ref_wav_path"])

    # trim silence
    try:
        y, sr = librosa.load(new_data["wav_path"], sr=None)
        _, (start, end) = librosa.effects.trim(
            y,
            top_db=TRIM_THRESHOLD_IN_DB,
            frame_length=TRIM_FRAME_SIZE,
            hop_length=TRIM_HOP_SIZE,
        )
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None
    
    new_data["start"] = start / sr
    new_data["end"] = end / sr

    # write spk
    new_data["spk"] = spk

    return new_data

def process_sample_wrapper(arg_tuple):
    args, texts, item = arg_tuple
    return process_sample(args, texts, item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_csv", type=str, required=True, help="original csv file"
    )
    parser.add_argument("--db_root", type=str, required=True, help="database root")
    parser.add_argument("--out", type=str, required=True, help="output file path")
    args = parser.parse_args()

    # read original csv file
    original_csv, _ = read_csv(args.original_csv, dict_reader=True)

    # read all text files
    # text is in jvs001/parallel100/transcripts_utf8.txt
    texts = {}
    for i in range(100):
        spk = f"jvs{i+1:03d}"
        spk_texts = {}
        for _set in ["parallel", "nonparallel"]:
            text_path = os.path.join(
                args.db_root, spk, SET_MAPPING[_set], "transcripts_utf8.txt"
            )
            with open(text_path, "r") as f:
                lines = f.read().splitlines()
            for line in lines:
                _id, text = line.split(":")
                spk_texts[_id] = text
        texts[spk] = spk_texts

    # put text into csv
    data = []
    with ProcessPoolExecutor() as executor:
        arg_tuples = [(args, texts, item) for item in tqdm(original_csv)]
        results = list(tqdm(executor.map(process_sample_wrapper, arg_tuples), total=len(arg_tuples)))

    for item in results:
        if item is None:
            continue
        data.append(item)

    # write to out
    write_csv(data, args.out)
