#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import jiwer
import librosa
import nue_asr
import numpy as np
import pyopenjtalk
import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from jatts.evaluate.dtw_based import calculate_mcd_f0
from jatts.utils import read_csv


def load_asr_model(device):
    """Load model"""
    model = nue_asr.load_model("rinna/nue-asr")
    tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")
    models = {"model": model, "tokenizer": tokenizer}
    return models


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    sentence = pyopenjtalk.g2p(sentence, kana=True)

    return sentence


def transcribe(model, device, wav):
    """Calculate score on one single waveform"""
    audio = librosa.util.pad_center(wav, size=len(wav) + 16000, mode="constant")
    transcription = nue_asr.transcribe(model["model"], model["tokenizer"], audio).text
    return transcription


def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def _calculate_asr_score(model, device, dataset, wavdir):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, item in enumerate(tqdm(dataset)):
        sample_id = item["sample_id"]
        groundtruth = item["original_text"]
        generated_wav_path = os.path.join(wavdir, sample_id + ".wav")

        # load waveform
        wav, _ = librosa.load(generated_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(
            groundtruth, transcription
        )

        ers[sample_id] = [
            c_result["cer"] * 100.0,
            w_result["wer"] * 100.0,
            norm_transcription,
            norm_groundtruth,
        ]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]

    # calculate over whole set
    def er(r):
        return (
            float(r["substitutions"] + r["deletions"] + r["insertions"])
            / float(r["substitutions"] + r["deletions"] + r["hits"])
            * 100.0
        )

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer


def _calculate_mcd_f0(dataset, wavdir, f0_all, results):
    for i, item in enumerate(dataset):
        sample_id = item["sample_id"]
        spk = sample_id.split("_")[0]
        f0min = f0_all[spk]["f0min"]
        f0max = f0_all[spk]["f0max"]

        # get ground truth target wav path
        gt_wav_path = item["wav_path"]
        generated_wav_path = os.path.join(wavdir, sample_id + ".wav")

        # read both converted and ground truth wav
        generated_wav, generated_fs = librosa.load(generated_wav_path, sr=None)
        gt_wav, gt_fs = librosa.load(gt_wav_path, sr=generated_fs)
        if generated_fs != gt_fs:
            generated_wav = torchaudio.transforms.Resample(generated_fs, gt_fs)(
                torch.from_numpy(generated_wav)
            ).numpy()

        # calculate MCD, F0RMSE, F0CORR and DDUR
        res = calculate_mcd_f0(generated_wav, gt_wav, gt_fs, f0min, f0max)

        results.append([sample_id, res])


def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument(
        "--wavdir", required=True, type=str, help="directory for converted waveforms"
    )
    parser.add_argument("--csv", type=str, required=True, help="path to csv file")
    parser.add_argument(
        "--f0_path", required=True, type=str, help="file storing f0 ranges"
    )
    parser.add_argument(
        "--n_jobs", default=10, type=int, help="number of parallel jobs"
    )
    return parser


def main():
    args = get_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = read_csv(args.csv, dict_reader=True)
    print("number of utterances = {}".format(len(dataset)))

    # load f0min and f0 max
    f0_all = {}
    with open(args.f0_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines[1:]:
            spk, gender, f0min, f0max = line.split(" ")
            f0_all[spk] = {
                "f0min": int(f0min),
                "f0max": int(f0max),
            }

    ##############################

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(asr_model, device, dataset, args.wavdir)

    ##############################

    print("Calculating SHEET scores...")

    # load model
    sheet_predictor = torch.hub.load(
        "unilight/sheet:v0.1.0", "default", trust_repo=True, force_reload=True
    )

    # calculate scores
    sheet_scores = {}
    for item in tqdm(dataset):
        sheet_scores[item["sample_id"]] = sheet_predictor.predict(
            wav_path=os.path.join(args.wavdir, item["sample_id"] + ".wav")
        )

    ##############################

    print("Calculating MCD and f0-related scores...")
    # Get and divide list
    file_lists = np.array_split(dataset, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        results = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(
                target=_calculate_mcd_f0,
                args=(f, args.wavdir, f0_all, results),
            )
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        sorted_results = sorted(results, key=lambda x: x[0])
        results = []
        for result in sorted_results:
            d = {k: v for k, v in result[1].items()}
            d["basename"] = result[0]
            d["CER"] = ers[result[0]][0]
            d["GT_TRANSCRIPTION"] = ers[result[0]][2]
            d["CV_TRANSCRIPTION"] = ers[result[0]][3]
            d["SHEET_SCORE"] = sheet_scores[result[0]]
            results.append(d)

    # utterance wise result
    for result in results:
        print(
            "{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} \t{} | {}".format(
                result["basename"],
                result["MCD"],
                result["F0RMSE"],
                result["F0CORR"],
                result["DDUR"],
                result["SHEET_SCORE"],
                result["CER"],
                result["GT_TRANSCRIPTION"],
                result["CV_TRANSCRIPTION"],
            )
        )

    # average result
    mMCD = np.mean(np.array([result["MCD"] for result in results]))
    mf0RMSE = np.mean(np.array([result["F0RMSE"] for result in results]))
    mf0CORR = np.mean(np.array([result["F0CORR"] for result in results]))
    mDDUR = np.mean(np.array([result["DDUR"] for result in results]))
    mSHEET_SCORE = np.mean(np.array([result["SHEET_SCORE"] for result in results]))
    mCER = cer

    print(
        "Mean MCD, f0RMSE, f0CORR, DDUR, SHEET_SCORE, CER: {:.2f} {:.2f} {:.3f} {:.3f} {:.2f} {:.1f}".format(
            mMCD, mf0RMSE, mf0CORR, mDDUR, mSHEET_SCORE, mCER
        )
    )


if __name__ == "__main__":
    main()
