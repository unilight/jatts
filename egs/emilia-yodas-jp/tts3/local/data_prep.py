#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation script for Emilia-YODAS JP.
"""

import argparse
import csv
import os
import librosa
import pyopenjtalk
from jatts.utils.utils import read_csv, write_csv
from tqdm import tqdm
from collections import defaultdict
import random

DURATION_MIN = 4.0

def read_all_files(dir_path):
    all_files = {}
    for _f in ["segments", "spk2utt", "text", "utt2spk", "wav.scp"]:
        file_path = os.path.join(dir_path, _f)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
            all_files[_f] = lines
    return all_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, required=True, help="database root")
    parser.add_argument(
        "--out", type=str, required=True, help="output csv path"
    )
    args = parser.parse_args()

    print("Read all.csv...")
    all_data, _ = read_csv(os.path.join(args.db_root, "all.csv"), dict_reader=True)

    # format of all.csv:
    # _id,dnsmos,duration,language,phone_count,speaker,text
    # JA_QjF3JfRxetE_W000000,2.5341,3.96,ja,13,JA_QjF3JfRxetE_SPEAKER_01,ワン!レディーゴー!
    # Group data by speaker
    speaker_data = defaultdict(list)
    for item in all_data:
        speaker = item["speaker"]
        speaker_data[speaker].append(item)

    data = []
    for item in tqdm(all_data):
        speaker = item["speaker"]
        sample_id = item["_id"]
        wav_path = os.path.join(args.db_root, "Emilia-YODAS", "ja", speaker, sample_id + ".wav")
        assert os.path.exists(wav_path), f"{wav_path} does not exist."

        # Filter candidates based on conditions
        valid_candidates = [
            candidate for candidate in speaker_data[speaker]
            if candidate["_id"] != sample_id and float(candidate["duration"]) > DURATION_MIN
        ]
        # Randomly pick one valid candidate
        if valid_candidates:
            prompt = random.choice(valid_candidates)
            new_data = {
                "sample_id": sample_id,
                "wav_path": wav_path,
                "spk": speaker,
                "original_text": item["text"],
                "phonemes": pyopenjtalk.g2p(item["text"]),
                "start": 0,
                "end": float(item["duration"]),
                "prompt_sample_id": prompt["_id"],
                "prompt_wav_path": os.path.join(args.db_root, "Emilia-YODAS", "ja", speaker, prompt["_id"] + ".wav"),
                "prompt_start": 0,
                "prompt_end": float(prompt["duration"]),
            }
            data.append(new_data)

    # Calculate statistics
    total_samples = len(data)
    total_speakers = len(set(item["spk"] for item in data))
    total_duration = sum(float(item["duration"]) for item in all_data) / 3600  # Convert seconds to hours

    # Print statistics
    print(f"Total number of samples: {total_samples}")
    print(f"Total number of speakers: {total_speakers}")
    print(f"Total duration in hours: {total_duration:.2f}")

    write_csv(
        sorted(data, key=lambda x:x["sample_id"]), args.out
    )
