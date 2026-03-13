#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Data preparation script for CSJ.
Note: lecture ID is basically speaker ID.

20250501: We will discard this setting
This setting is to randomly select a segment from the same lecture ID
but we need to know the transcription of the segment, which is impossible
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

TRAIN_SUBDIR = "train_nodup"
DEV_SUBDIR = "train_dev"
TEST_SUBDIRS = ["eval1", "eval2", "eval3"]

DURATION_MIN = 4.0
DURATION_MAX = 10.0
PROMPT_DURATION = 10.0


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
    parser.add_argument(
        "--originals_dir",
        type=str,
        required=True,
        help="directory for the original files from ESPnet",
    )
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

    ##############
    # read train #
    ##############
    print("Reading train set...")
    total_duration = 0.0
    train_all_files = read_all_files(os.path.join(args.originals_dir, "train_nodup"))
    train_data = defaultdict(dict)
    train_wavs = defaultdict(dict)
    train_spk2utt = {}
    # read spk2utt
    for line in tqdm(train_all_files["spk2utt"], desc="Reading spk2utt"):
        parts = line.split(" ")
        train_spk2utt[parts[0]] = parts[1:]
    # read wav.scp first, because it contains only the lecture id
    for line in tqdm(train_all_files["wav.scp"], desc="Reading wav.scp"):
        lecture_id, _, wav_path, _ = line.split(" ")
        train_wavs[lecture_id]["wav_path"] = wav_path
        train_wavs[lecture_id]["start"] = 99999.0
        train_wavs[lecture_id]["end"] = -1.0
    # read segments
    for line in tqdm(train_all_files["segments"], desc="Reading segments"):
        sample_id, lecture_id, start, end = line.split(" ")
        start, end = float(start), float(end)
        duration = end - start
        total_duration += duration
        item = {
            "sample_id": sample_id,
            "lecture_id": lecture_id,
            "wav_path": train_wavs[lecture_id]["wav_path"],
            "start": start,
            "end": end,
        }
        train_data[sample_id] = item
        # update start and end in train_wavs
        if start < train_wavs[lecture_id]["start"]:
            train_wavs[lecture_id]["start"] = start
        if end > train_wavs[lecture_id]["end"]:
            train_wavs[lecture_id]["end"] = end
    # read text and assign prompt
    for line in tqdm(train_all_files["text"], desc="Reading text"):
        sample_id, original_text = line.split(" ", 1)
        if sample_id in train_data:
            phonemes = pyopenjtalk.g2p(original_text)
            train_data[sample_id]["original_text"] = original_text
            train_data[sample_id]["phonemes"] = phonemes

            # add prompt
            lecture_id = train_data[sample_id]["lecture_id"]
            train_data[sample_id]["prompt_wav_path"] = train_wavs[lecture_id]["wav_path"]
            start, end = train_wavs[lecture_id]["start"], train_wavs[lecture_id]["end"]
            # Ensure the prompt duration is at least PROMPT_DURATION
            assert end - start >= PROMPT_DURATION, f"{lecture_id} is less than {PROMPT_DURATION} seconds."
            # Randomly select a start time within the valid range
            prompt_start = random.uniform(start, end - PROMPT_DURATION)
            train_data[sample_id]["prompt_start"] = prompt_start
            train_data[sample_id]["prompt_end"] = prompt_start + PROMPT_DURATION
    # Print total duration in hours
    print(f"Total duration: {total_duration / 3600:.2f} hours")
    write_csv(
        sorted(list(train_data.values()), key=lambda x: x["sample_id"]), os.path.join(args.outdir, args.train_set + ".csv")
    )

    ############
    # read dev #
    ############
    print("Reading dev set...")
    dev_all_files = read_all_files(os.path.join(args.originals_dir, "train_dev"))
    dev_data = defaultdict(dict)
    dev_wavs = {}
    dev_spk2utt = {}
    # read wav.scp first, because it contains only the lecture id
    for line in tqdm(dev_all_files["wav.scp"], desc="Reading wav.scp"):
        lecture_id, _, wav_path, _ = line.split(" ")
        dev_wavs[lecture_id] = wav_path
    # read segments
    for line in tqdm(dev_all_files["segments"], desc="Reading segments"):
        sample_id, lecture_id, start, end = line.split(" ")
        duration = float(end) - float(start)
        if duration >= DURATION_MIN and duration <= DURATION_MAX:
            item = {
                "sample_id": sample_id,
                "lecture_id": lecture_id,
                "wav_path": dev_wavs[lecture_id],
                "start": float(start),
                "end": float(end),
            }
            dev_data[sample_id] = item
    # read spk2utt
    for line in tqdm(dev_all_files["spk2utt"], desc="Reading spk2utt"):
        parts = line.split(" ")
        dev_spk2utt[parts[0]] = [sid for sid in parts[1:] if sid in dev_data]
    # read text
    for line in tqdm(dev_all_files["text"], desc="Reading text"):
        sample_id, original_text = line.split(" ", 1)
        if sample_id in dev_data:
            phonemes = pyopenjtalk.g2p(original_text)
            dev_data[sample_id]["original_text"] = original_text
            dev_data[sample_id]["phonemes"] = phonemes

            # Get all sample_ids for the same lecture_id
            candidate_sample_ids = dev_spk2utt[dev_data[sample_id]["lecture_id"]]
            # Filter candidates based on conditions
            valid_candidates = [
                sid for sid in candidate_sample_ids
                if sid != sample_id and DURATION_MIN <= (dev_data[sid]["end"] - dev_data[sid]["start"]) <= DURATION_MAX
            ]
            # Randomly pick one valid candidate
            if valid_candidates:
                prompt_sample_id = random.choice(valid_candidates)
                dev_data[sample_id]["prompt_sample_id"] = prompt_sample_id
                dev_data[sample_id]["prompt_wav_path"] = dev_data[prompt_sample_id]["wav_path"]
                dev_data[sample_id]["prompt_start"] = dev_data[prompt_sample_id]["start"]
                dev_data[sample_id]["prompt_end"] = dev_data[prompt_sample_id]["end"]
                dev_data[sample_id]["prompt_original_text"] = dev_data[prompt_sample_id]["original_text"]
                dev_data[sample_id]["prompt_phonemes"] = dev_data[prompt_sample_id]["phonemes"]
    write_csv(
        sorted(list(dev_data.values()), key=lambda x:x["sample_id"]), os.path.join(args.outdir, args.dev_set + ".csv")
    )


    #############
    # read test #
    #############
    test_data = defaultdict(dict)

    for _set in TEST_SUBDIRS:
        _set_wavs = {}
        _set_segments = {}
        _set_texts = {}
        print(f"Reading {_set} set...")
        _set_all_files = read_all_files(os.path.join(args.originals_dir, _set))
        # read wav.scp first, because it contains only the lecture id
        for line in tqdm(_set_all_files["wav.scp"], desc="Reading wav.scp"):
            lecture_id, _, wav_path, _ = line.split(" ")
            _set_wavs[lecture_id] = wav_path
        # read text
        for line in tqdm(_set_all_files["text"], desc="Reading text"):
            sample_id, original_text = line.split(" ", 1)
            phonemes = pyopenjtalk.g2p(original_text)
            _set_texts[sample_id] = {
                "original_text": original_text,
                "phonemes": phonemes,
            }
        # read segments, and put everything (including wav_path and texts) in _set_segments
        for line in tqdm(_set_all_files["segments"], desc="Reading segments"):
            sample_id, lecture_id, start, end = line.split(" ")
            item = {
                "sample_id": sample_id,
                "lecture_id": lecture_id,
                "wav_path": _set_wavs[lecture_id],
                "start": float(start),
                "end": float(end),
                "phonemes": _set_texts[sample_id]["phonemes"],
                "original_text": _set_texts[sample_id]["original_text"],
            }
            _set_segments[sample_id] = item
        # read spk2utt
        for line in tqdm(_set_all_files["spk2utt"], desc="Reading wav.scp"):
            parts = line.split(" ")
            lecture_id = parts[0]
            _filtered_sample_ids = []
            for sample_id in parts[1:]:
                duration = (
                    _set_segments[sample_id]["end"] - _set_segments[sample_id]["start"]
                )
                if duration >= DURATION_MIN and duration <= DURATION_MAX:
                    _filtered_sample_ids.append(sample_id)
            # Shuffle _filtered_sample_ids before splitting into two halves
            # Then, split the list into two halves. If the length is odd, the second half will have one more element.
            random.shuffle(_filtered_sample_ids)
            half = len(_filtered_sample_ids) // 2
            first_half = _filtered_sample_ids[:half]
            second_half = _filtered_sample_ids[half:]

            # Assign a random sample_id from the second half to each sample_id in the first half
            random.shuffle(second_half)
            for idx, sample_id in enumerate(first_half):
                if idx < len(second_half):
                    test_data[sample_id] = _set_segments[sample_id]
                    prompt_sample_id = second_half[idx]
                    test_data[sample_id]["prompt_sample_id"] = prompt_sample_id
                    test_data[sample_id]["prompt_wav_path"] = _set_segments[
                        prompt_sample_id
                    ]["wav_path"]
                    test_data[sample_id]["prompt_start"] = _set_segments[
                        prompt_sample_id
                    ]["start"]
                    test_data[sample_id]["prompt_end"] = _set_segments[
                        prompt_sample_id
                    ]["end"]
                    test_data[sample_id]["prompt_original_text"] = _set_texts[
                        prompt_sample_id
                    ]["original_text"]
                    test_data[sample_id]["prompt_phonemes"] = _set_texts[
                        prompt_sample_id
                    ]["phonemes"]
    write_csv(
        sorted(list(test_data.values()), key=lambda x:x["sample_id"]), os.path.join(args.outdir, args.test_set + ".csv")
    )
