#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import csv
import math
import os
from collections import defaultdict

import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

from jatts.utils.utils import read_csv

BIGN = 10000


def calculate_frames(wav_path, phoneme_intervals, hop_size, fs, frame_length):
    frame_shift = hop_size / fs
    # Step 1: Calculate initial frame counts and total frames
    frame_counts = []
    total_frames = 0

    for start, end, phoneme in phoneme_intervals:
        duration = end - start
        frames = duration / frame_shift

        rounded_frames = int(frames)

        frame_counts.append((start, end, phoneme, rounded_frames))
        total_frames += rounded_frames

    # Step 2: Adjust rounding to ensure total frames match expected count
    n_samples = len(
        librosa.load(
            wav_path,
            sr=fs,
            offset=float(phoneme_intervals[0][0]),
            duration=float(phoneme_intervals[-1][1]) - float(phoneme_intervals[0][0]),
        )[0]
    )
    if n_samples % hop_size == 0:
        expected_total_frames = int(n_samples / hop_size) + 1
    else:
        expected_total_frames = math.floor(n_samples / hop_size) + 1

    adjustment = expected_total_frames - total_frames
    assert adjustment >= 0, f"expected total frames ({expected_total_frames}) is smaller than total frames ({total_frames})"
    if adjustment > 0:
        frame_differences = [
            frames - (end - start) / frame_shift
            for start, end, phoneme, frames in frame_counts
        ]
        adjustment_order = sorted(
            range(len(frame_differences)),
            key=lambda i: abs(frame_differences[i]),
            reverse=True,
        )

        for i in adjustment_order:
            if adjustment == 0:
                break

            start, end, phoneme, frames = frame_counts[i]

            if adjustment > 0:
                frame_counts[i] = (start, end, phoneme, frames + 1)
                adjustment -= 1
            elif (
                adjustment < 0 and frames > 1
            ):  # Ensure at least one frame remains per phoneme
                frame_counts[i] = (start, end, phoneme, frames - 1)
                adjustment += 1

    return frame_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--juliusdir", type=str, required=True, help="julius segmentation results dir"
    )
    parser.add_argument(
        "--original_csv", type=str, required=True, help="original wavscp file"
    )
    parser.add_argument(
        "--conf", type=str, required=True, help="config file (to get fs and shift)"
    )
    parser.add_argument("--out", type=str, required=True, help="out wavscp file")

    args = parser.parse_args()

    # load config
    with open(args.conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    n_shift = config["hop_size"]
    fs = config["sampling_rate"]
    frame_length = config["fft_size"]

    # read csv
    original_csv, _ = read_csv(args.original_csv, dict_reader=True)

    # loop
    data = []
    for item in tqdm(original_csv):
        # utt_id looks like this: jvs091_parallel_VOICEACTRESS100_091
        spk, _ = item["sample_id"].split("_", 1)
        lab_path = os.path.join(args.juliusdir, item["sample_id"] + ".lab")
        with open(lab_path, "r") as f:
            lines = f.read().splitlines()
        # skip if segmentation fail
        if len(lines) < 1:
            continue

        phns = []
        phn_intervals = []
        for i, line in enumerate(lines):
            start, end, phn = line.split(" ")
            if phn == "silB":
                utt_start = lines[i + 1].split(" ")[0]
                continue
            elif phn == "silE":
                utt_end = lines[i - 1].split(" ")[1]
                continue
            phn_intervals.append([float(start), float(end), phn])
            phns.append(phn)

        phn_frames = calculate_frames(
            item["wav_path"], phn_intervals, n_shift, fs, frame_length
        )
        durations = [str(d) for _, _, _, d in phn_frames]

        new_item = {k: v for k, v in item.items()}

        new_item["start"] = utt_start
        new_item["end"] = utt_end
        new_item["phonemes"] = " ".join(phns)
        new_item["durations"] = " ".join(durations)

        data.append(new_item)

    # write to out
    fieldnames = list(data[0].keys())
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in data:
            writer.writerow(line)
