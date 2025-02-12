#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import logging
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
from jatts.evaluate.dtw_based import calculate_mcd_f0
from jatts.modules.feature_extract.spkemb_speechbrain import SpeechBrainSpkEmbExtractor
from jatts.utils import read_csv
from prettytable import PrettyTable
from tqdm import tqdm


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
        spk = item["spk"]
        f0min = f0_all[spk]["f0min"]
        f0max = f0_all[spk]["f0max"]

        # get ground truth target wav path
        gt_wav_path = item["wav_path"]
        generated_wav_path = os.path.join(wavdir, sample_id + ".wav")

        # get start and end
        if "start" in item and "end" in item:
            offset = float(item["start"])
            duration = float(item["end"]) - float(item["start"])
        else:
            offset = 0.0
            duration = None

        # read both converted and ground truth wav
        generated_wav, generated_fs = librosa.load(generated_wav_path, sr=None)
        gt_wav, gt_fs = librosa.load(
            gt_wav_path, sr=generated_fs, offset=offset, duration=duration
        )
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
        "--metrics",
        nargs="+",
        required=True,
        default=["mcd", "sheet", "asr"],
        help="metrics to evaluate",
    )
    parser.add_argument(
        "--n_jobs", default=10, type=int, help="number of parallel jobs"
    )
    return parser


def main():
    args = get_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = read_csv(args.csv, dict_reader=True)
    logging.info("number of utterances = {}".format(len(dataset)))

    # load f0min and f0 max
    with open(args.f0_path, "r") as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)

    # start evaluation
    results = []
    table = PrettyTable()
    table.add_column("Sample ID", [item["sample_id"] for item in dataset])
    to_print = []
    for metric in args.metrics:
        if metric == "asr":
            logging.info("Calculating ASR-based score...")

            # load ASR model
            asr_model = load_asr_model(device)

            # calculate error rates
            ers, cer, wer = _calculate_asr_score(
                asr_model, device, dataset, args.wavdir
            )
            mCER = cer
            to_print.append(f"CER = {mCER:.1f}")

            table.add_column("CER", [ers[item["sample_id"]][0] for item in dataset])
            table.add_column("GT Text", [ers[item["sample_id"]][2] for item in dataset])
            table.add_column(
                "Transcription", [ers[item["sample_id"]][3] for item in dataset]
            )
            table.custom_format["CER"] = lambda f, v: f"{v:.1f}"
            table.align["CER"] = "r"
            table.align["GT Text"] = "l"
            table.align["Transcription"] = "l"

        if metric == "spkemb":
            logging.info("Calculating speaker embedding cosine similarity...")

            # load speaker embedding model
            spkemb_extractor = SpeechBrainSpkEmbExtractor(device)

            # calculate scores
            spkemb_sim_scores = {}
            for item in tqdm(dataset):
                generated_wav_path = os.path.join(
                    args.wavdir, item["sample_id"] + ".wav"
                )
                gt_wav_path = item["ref_wav_path"]
                generated_wav_spkemb = spkemb_extractor.forward(generated_wav_path)
                gt_wav_spkemb = spkemb_extractor.forward(gt_wav_path)
                sim = np.inner(generated_wav_spkemb, gt_wav_spkemb) / (
                    np.linalg.norm(generated_wav_spkemb) * np.linalg.norm(gt_wav_spkemb)
                )
                spkemb_sim_scores[item["sample_id"]] = sim
            mSPKEMB_SIM_SCORE = np.mean(
                np.array([v for v in spkemb_sim_scores.values()])
            )
            to_print.append(f"SPKEMB SIM = {mSPKEMB_SIM_SCORE:.3f}")

            table.add_column(
                "SPKEMB SIM", [spkemb_sim_scores[item["sample_id"]] for item in dataset]
            )
            table.custom_format["SPKEMB SIM"] = lambda f, v: f"{v:.3f}"

        if metric == "sheet":

            logging.info("Calculating SHEET scores...")

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
            mSHEET_SCORE = np.mean(np.array([v for v in sheet_scores.values()]))
            to_print.append(f"SHEET SCORE = {mSHEET_SCORE:.3f}")

            table.add_column(
                "SHEET Score", [sheet_scores[item["sample_id"]] for item in dataset]
            )
            table.custom_format["SHEET Score"] = lambda f, v: f"{v:.2f}"

        if metric == "mcd":
            logging.info("Calculating MCD and f0-related scores...")

            # Get and divide list
            file_lists = np.array_split(dataset, args.n_jobs)
            file_lists = [f_list.tolist() for f_list in file_lists]

            # multi processing
            with mp.Manager() as manager:
                _results = manager.list()
                processes = []
                for f in file_lists:
                    p = mp.Process(
                        target=_calculate_mcd_f0,
                        args=(f, args.wavdir, f0_all, _results),
                    )
                    p.start()
                    processes.append(p)

                # wait for all process
                for p in processes:
                    p.join()

                mcd_results = {
                    _result[0]: {k: v for k, v in _result[1].items()}
                    for _result in _results
                }
                mMCD = np.mean(np.array([v["MCD"] for v in mcd_results.values()]))
                mf0RMSE = np.mean(np.array([v["F0RMSE"] for v in mcd_results.values()]))
                mf0CORR = np.mean(np.array([v["F0CORR"] for v in mcd_results.values()]))
                mDDUR = np.mean(np.array([v["DDUR"] for v in mcd_results.values()]))
                to_print.append(f"MCD = {mMCD:.2f}")
                to_print.append(f"F0RMSE = {mf0RMSE:.3f}")
                to_print.append(f"F0CORR = {mf0CORR:.3f}")
                to_print.append(f"DDUR = {mDDUR:.3f}")

                table.add_column(
                    "MCD", [mcd_results[item["sample_id"]]["MCD"] for item in dataset]
                )
                table.add_column(
                    "F0RMSE",
                    [mcd_results[item["sample_id"]]["F0RMSE"] for item in dataset],
                )
                table.add_column(
                    "F0CORR",
                    [mcd_results[item["sample_id"]]["F0CORR"] for item in dataset],
                )
                table.add_column(
                    "DDUR", [mcd_results[item["sample_id"]]["DDUR"] for item in dataset]
                )
                table.custom_format["MCD"] = lambda f, v: f"{v:.2f}"
                table.custom_format["F0RMSE"] = lambda f, v: f"{v:.3f}"
                table.custom_format["F0CORR"] = lambda f, v: f"{v:.3f}"
                table.custom_format["DDUR"] = lambda f, v: f"{v:.3f}"

    logging.info("Mean " + "; ".join(to_print))
    table.sortby = "Sample ID"
    print(table)


if __name__ == "__main__":
    main()
