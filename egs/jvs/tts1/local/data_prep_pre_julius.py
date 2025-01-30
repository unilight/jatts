#!/usr/bin/env python3

import argparse
import csv
import os

from jatts.utils.utils import read_csv, write_csv

SET_MAPPING = {"parallel": "parallel100", "nonparallel": "nonpara30"}

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
    for item in original_csv:
        # sample_id looks like jvs001_parallel_VOICEACTRESS100_001
        sample_id = item["sample_id"]
        spk, _set, _id = sample_id.split("_", 2)

        new_data = {k: v for k, v in item.items()}

        # fill in text
        new_data["original_text"] = texts[spk][_id]

        # override wav_path to absolute path
        new_data["wav_path"] = os.path.join(args.db_root, item["wav_path"])

        # write spk
        new_data["spk"] = spk

        data.append(new_data)

    # write to out
    write_csv(data, args.out)
