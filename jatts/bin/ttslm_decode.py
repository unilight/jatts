#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained TTS model."""

import argparse
import logging
import os
import time

import jatts.models
import numpy as np
import soundfile as sf
import torch
import yaml
from jatts.datasets.tts_dataset import TTSDataset
from jatts.modules.feature_extract.encodec import EnCodec
from jatts.utils import read_hdf5
from jatts.utils.plot import plot_1d, plot_attention, plot_generated_and_ref_2d
from jatts.vocoder import Vocoder
from tqdm import tqdm
from pathlib import Path

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
        "--ar-checkpoint",
        type=str,
        required=True,
        help="ar checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--nar-checkpoint",
        type=str,
        required=True,
        help="nar checkpoint file to be loaded.",
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

    # Load AR config
    if args.config is None:
        ar_dirname = os.path.dirname(args.ar_checkpoint)
        ar_config_path = os.path.join(ar_dirname, "config.yml")
    else:
        ar_config_path = args.config
    with open(ar_config_path) as f:
        ar_config = yaml.load(f, Loader=yaml.Loader)

    # Load NAR config
    nar_dirname = os.path.dirname(args.nar_checkpoint)
    nar_config_path = os.path.join(nar_dirname, "config.yml")
    with open(nar_config_path) as f:
        nar_config = yaml.load(f, Loader=yaml.Loader)

    # Merge configs
    config = {**ar_config, **nar_config}
    config.update(vars(args))

    # get dataset
    dataset = TTSDataset(
        csv_path=args.csv,
        stats_path=None,
        feat_list=config["feat_list"],
        token_list_path=args.token_list,
        token_column=args.token_column,
        is_inference=True,
        prompt_path=config.get("prompt_path", None),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"Dataset size = {len(dataset)}.")

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get AR model and load parameters
    model_class = getattr(jatts.models, "ValleAR")
    ar_model = model_class(**ar_config["model_params"])
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location="cpu")["model"])
    ar_model = ar_model.eval().to(device)
    logging.info(f"Loaded AR model parameters from {args.ar_checkpoint}.")

    # get NAR model and load parameters
    nar_model_class = getattr(jatts.models, "ValleNAR")
    nar_model = nar_model_class(**nar_config["model_params"])
    nar_model.load_state_dict(torch.load(args.nar_checkpoint, map_location="cpu")["model"])
    nar_model = nar_model.eval().to(device)
    logging.info(f"Loaded NAR model parameters from {args.nar_checkpoint}.")

    # load EnCodec decoder
    extractor = EnCodec()
    
    # start generation
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for _, item in enumerate(pbar, 1):
            sample_id = item["sample_id"]

            # prepare input
            x = torch.tensor(item["token_indices"], dtype=torch.long).to(device)
            prompts = torch.tensor(item["prompts"], dtype=torch.long).to(device)
            # model forward
            start_time = time.time()

            # AR model inference
            ar_codes = ar_model([x], [prompts], max_steps=config.get("max_ar_steps", 1000))
            ar_codes = [code.unsqueeze(-1) for code in ar_codes]

            # NAR model inference
            nar_codes = nar_model([x], [prompts], resps_list=ar_codes)
            outs = nar_codes[0]

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            #plot_generated_and_ref_2d(
            #    outs.cpu().numpy(),
            #    config["outdir"] + f"/outs/{sample_id}.png",
            #    origin="lower",
            #)

            if not os.path.exists(os.path.join(config["outdir"], "wav")):
                os.makedirs(os.path.join(config["outdir"], "wav"), exist_ok=True)

            extractor.decode_to_file(
                resps=outs,
                path=Path(os.path.join(config["outdir"], "wav", f"{sample_id}.wav")),
            )


            ## when decoding dev set, for debugging purpose, synthesize analysis-synthesis voice
            #if "feat_path" in item:
            #    if not os.path.exists(os.path.join(config["outdir"], "wav_anasyn")):
            #        os.makedirs(
            #            os.path.join(config["outdir"], "wav_anasyn"), exist_ok=True
            #        )

            #    mel = torch.Tensor(read_hdf5(item["feat_path"], "mel"))
            #    mel = (mel - stats["mean"]) / stats["scale"]
            #    mel = mel.to(outs.device)

            #    y, sr = vocoder.decode(mel)
            #    sf.write(
            #        os.path.join(config["outdir"], "wav_anasyn", f"{sample_id}.wav"),
            #        y.cpu().numpy(),
            #        sr,
            #        "PCM_16",
            #    )


if __name__ == "__main__":
    main()
