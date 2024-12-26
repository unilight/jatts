#!/usr/bin/env python3

import argparse
import os
import librosa
from tqdm import tqdm
import soundfile as sf

import pyopenjtalk
import jaconv

from jatts.utils.utils import read_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="csv file")
    parser.add_argument("--outdir", type=str, required=True, help="output dir")
    args = parser.parse_args()

    data, _ = read_csv(args.csv, dict_reader=True)

    for item in tqdm(data):
        target_wav_path = os.path.join(args.outdir, item["sample_id"] + ".wav")
        if not os.path.exists(target_wav_path):
            data, sr = librosa.load(item["wav_path"], sr=16000)
            sf.write(target_wav_path, data, 16000, "PCM_16")

        with open(os.path.join(args.outdir, item["sample_id"] + ".txt"), "w") as f:
            g2p_result = pyopenjtalk.g2p(item["original_text"], kana=True)
            hira = jaconv.kata2hira(g2p_result)
            normalized_g2p_result = hira.replace("。", "").replace("、", " sp ")
            f.write(normalized_g2p_result)