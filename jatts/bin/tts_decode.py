#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained TTS model."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

import jatts.models
from jatts.datasets.tts_dataset import TTSDataset
from jatts.utils import read_hdf5
from jatts.utils.plot import plot_1d, plot_attention, plot_generated_and_ref_2d
from jatts.vocoder import Vocoder

# from jatts.vocoder.s3prl_feat2wav import S3PRL_Feat2Wav
# from jatts.vocoder.griffin_lim import Spectrogram2Waveform
# from jatts.vocoder.encodec import EnCodec_decoder


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode with trained TTS model " "(See detail in bin/tts_decode.py)."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help=("training csv file. "),
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="stats file for denormalization.",
    )
    parser.add_argument(
        "--token-list",
        type=str,
        required=True,
        help="a text mapping int-id to token",
    )
    parser.add_argument(
        "--token-column",
        type=str,
        required=True,
        help="which column to use as the text input in the csv file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # get dataset
    dataset = TTSDataset(
        csv_path=args.csv,
        stats_path=args.stats,
        feat_list=config["feat_list"],
        token_list_path=args.token_list,
        token_column=args.token_column,
        is_inference=True,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"Dataset size = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get model and load parameters
    model_class = getattr(jatts.models, config["model_type"])
    model = model_class(**config["model_params"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # load speaker encoder if needed
    if "spkemb" in config["feat_list"]:
        if config["spkemb_extract_type"] == "speechbrain":
            from jatts.modules.feature_extract.spkemb_speechbrain import (
                SpeechBrainSpkEmbExtractor,
            )

            spkemb_extractor = SpeechBrainSpkEmbExtractor()
        else:
            raise ValueError(
                f"Unsupported speaker embedding extractor type: {config['spkemb_extract_type']}"
            )
            exit(1)

    # load output stats for denormalization
    out_feat_type = config["out_feat_type"]
    stats = {
        "mean": read_hdf5(args.stats, f"{out_feat_type}_mean"),
        "scale": read_hdf5(args.stats, f"{out_feat_type}_scale"),
    }

    # load vocoder
    if config.get("vocoder", False):
        vocoder_type = config["vocoder"].get("vocoder_type", "")
        if vocoder_type == "s3prl_vc":
            vocoder = S3PRL_Feat2Wav(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                stats,  # this is used to denormalized the converted features,
                device,
            )
        elif vocoder_type == "encodec":
            vocoder = EnCodec_decoder(
                stats,  # this is used to denormalized the converted features,
                device,
            )
        else:
            vocoder = Vocoder(
                config["vocoder"]["checkpoint"],
                config["vocoder"]["config"],
                config["vocoder"]["stats"],
                device,
                trg_stats=stats,  # this is used to denormalized the converted features,
            )
    else:
        vocoder = Spectrogram2Waveform(
            stats=config["stats"],
            n_fft=config["fft_size"],
            n_shift=config["hop_size"],
            fs=config["sampling_rate"],
            n_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            griffin_lim_iters=64,
        )

    # start generation
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for _, item in enumerate(pbar, 1):
            sample_id = item["sample_id"]

            # prepare input
            x = torch.tensor(item["token_indices"], dtype=torch.long).to(device)

            # model forward
            start_time = time.time()
            if "spkemb" in config["feat_list"]:
                spkemb = torch.tensor(
                    spkemb_extractor.forward(item["ref_wav_path"])
                ).to(device)
            else:
                spkemb = None
            ret = model.inference(x, spembs=spkemb)
            outs = ret["feat_gen"]
            d_outs = ret["duration"]

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            plot_generated_and_ref_2d(
                outs.cpu().numpy(),
                config["outdir"] + f"/outs/{sample_id}.png",
                origin="lower",
            )

            if not os.path.exists(os.path.join(config["outdir"], "wav")):
                os.makedirs(os.path.join(config["outdir"], "wav"), exist_ok=True)

            y, sr = vocoder.decode(outs)
            sf.write(
                os.path.join(config["outdir"], "wav", f"{sample_id}.wav"),
                y.cpu().numpy(),
                sr,
                "PCM_16",
            )


if __name__ == "__main__":
    main()
