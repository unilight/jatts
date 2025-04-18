#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from jatts.modules.feature_extract.dio import Dio
from jatts.modules.feature_extract.energy import Energy
from jatts.modules.feature_extract.mel import logmelfilterbank
from jatts.modules.feature_extract.spkemb_speechbrain import SpeechBrainSpkEmbExtractor
from jatts.modules.feature_extract.encodec import EnCodec
from jatts.utils import read_csv, write_csv, write_hdf5, read_audio
from tqdm import tqdm

# from s3prl.nn import Featurizer
# import s3prl_vc.models
# from s3prl_vc.upstream.interface import get_upstream





def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="csv file.",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--f0_path", default=None, type=str, help="file storing f0 ranges"
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

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # read dataset
    dataset, _ = read_csv(args.csv, dict_reader=True)

    # load f0min and f0 max if given
    if args.f0_path is not None:
        with open(args.f0_path, "r") as f:
            f0_all = yaml.load(f, Loader=yaml.FullLoader)
    else:
        f0_all = None

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # load upstream extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractors = {}
    for feat_type in config.get("feat_list", ["mel"]):
        if feat_type == "mel":
            extractor = {}
        elif feat_type == "pitch":
            if config["pitch_extract_type"] == "dio":
                extractor = Dio(
                    fs=config["sampling_rate"],
                    n_fft=config["fft_size"],
                    hop_length=config["hop_size"],
                    use_token_averaged_f0=True,
                    use_continuous_f0=True,
                    use_log_f0=True,
                    reduction_factor=config["model_params"]["reduction_factor"],
                )
            else:
                raise ValueError(
                    f"Unsupported pitch extractor type: {config['pitch_extract_type']}"
                )
                exit(1)
        elif feat_type == "energy":
            if config["energy_extract_type"] == "energy":
                extractor = Energy(
                    fs=config["sampling_rate"],
                    n_fft=config["fft_size"],
                    win_length=config["win_length"],
                    hop_length=config["hop_size"],
                    window=config["window"],
                    use_token_averaged_energy=True,
                    reduction_factor=config["model_params"]["reduction_factor"],
                )
            else:
                raise ValueError(
                    f"Unsupported energy extractor type: {config['energy_extract_type']}"
                )
                exit(1)
        elif feat_type == "spkemb":
            if config["spkemb_extract_type"] == "speechbrain":
                extractor = SpeechBrainSpkEmbExtractor()
            else:
                raise ValueError(
                    f"Unsupported speaker embedding extractor type: {config['spkemb_extract_type']}"
                )
                exit(1)

        elif feat_type == "encodec":
            extractor = EnCodec()
        # else:
        #     checkpoint = config["feat_list"][feat_type]["checkpoint"]
        #     upstream_model = get_upstream(feat_type).to(device)
        #     upstream_model.eval()
        #     upstream_featurizer = Featurizer(upstream_model).to(device)
        #     upstream_featurizer.load_state_dict(
        #         torch.load(checkpoint, map_location="cpu")["featurizer"]
        #     )
        #     upstream_featurizer.eval()
        #     logging.info(f"Loaded {feat_type} extractor parameters from {checkpoint}.")

        #     extractor = {"model": upstream_model, "featurizer": upstream_featurizer}

        extractors[feat_type] = extractor

    if "prompt_feat_list" in config:
        prompt_extractors = {}
        for feat_type in config["prompt_feat_list"]:
            if feat_type == "encodec":
                extractor = EnCodec()
            prompt_extractors[feat_type] = extractor
    else:
        prompt_extractors = None

    # process each data
    new_datas = []
    for item in tqdm(dataset):
        utt_id = item["sample_id"]

        # read main audio
        audio = read_audio(
            wav_path=item["wav_path"],
            sr=config["sampling_rate"],
            start=item.get("start", None),
            end=item.get("end", None),
            global_gain_scale=config["global_gain_scale"],
        )
        if audio is None:
            logging.warn(f"{wav_path} returns None. Skip.")
            continue

        # save waveform
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "wave",
            audio.astype(np.float32),
        )

        # read prompt audio if exists
        if "prompt_wav_path" in item:
            prompt_audio = read_audio(
                wav_path=item["prompt_wav_path"],
                sr=config["sampling_rate"],
                start=item.get("prompt_start", None),
                end=item.get("prompt_end", None),
                global_gain_scale=config["global_gain_scale"],
            )

            # save waveform
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "prompt_wave",
                prompt_audio.astype(np.float32),
            )
        else:
            prompt_audio = None

        # get phoneme-wise durations
        if "durations" in item:
            phoneme_wise_durations = [int(d) for d in item["durations"].split(" ")]
            total_duration = sum(phoneme_wise_durations)
        else:
            phoneme_wise_durations = None
            total_duration = None

        # extract and save feature
        feat_path = os.path.join(args.dumpdir, f"{utt_id}.h5")
        for feat_type, extractor in extractors.items():
            if feat_type == "mel":
                feat = logmelfilterbank(
                    audio,
                    sampling_rate=config["sampling_rate"],
                    hop_size=config["hop_size"],
                    fft_size=config["fft_size"],
                    win_length=config["win_length"],
                    window=config["window"],
                    num_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                    # keep compatibility
                    log_base=config.get("log_base", 10.0),
                )  # [n_frames, n_dim]

                # check duration
                if phoneme_wise_durations is not None:
                    assert (
                        total_duration == feat.shape[0]
                    ), f"{utt_id}: mel duration {feat.shape[0]} and phoneme durations {total_duration} don't match ."

            elif feat_type == "pitch":
                assert phoneme_wise_durations is not None
                if f0_all is not None:
                    f0min = f0_all[item["spk"]]["f0min"]
                    f0max = f0_all[item["spk"]]["f0max"]
                else:
                    f0min = config["pitch_extract_f0min"]
                    f0max = config["pitch_extract_f0max"]
                feat = extractor.forward(
                    audio,
                    f0min=f0min,
                    f0max=f0max,
                    feat_length=total_duration,
                    durations=np.array(phoneme_wise_durations),
                )
            elif feat_type == "energy":
                assert phoneme_wise_durations is not None
                feat = extractor.forward(
                    audio,
                    feat_length=total_duration,
                    durations=np.array(phoneme_wise_durations),
                )
            elif feat_type == "spkemb":
                feat = extractor.forward(item["wav_path"])
            elif feat_type == "encodec":
                feat = extractor.encode(audio, config["sampling_rate"], device)
                feat = feat.squeeze(0).cpu().numpy()  # q, t
            else:
                logging.info(f"Not supported feature type {feat_type}. Skip.")
                continue

            # else:
            #     with torch.no_grad():
            #         xs = torch.from_numpy(x).unsqueeze(0).float().to(device)
            #         ilens = torch.LongTensor([x.shape[0]]).to(device)

            #         all_hs, all_hlens = extractors[feat_type]["model"](xs, ilens)
            #         hs, _ = extractors[feat_type]["featurizer"](all_hs, all_hlens)
            #         feat = hs[0].cpu().numpy()

            write_hdf5(
                feat_path,
                feat_type,
                feat.astype(np.float32),
            )
        if prompt_extractors is not None:
            assert prompt_audio is not None, "Prompt audio is not given."
            for feat_type, extractor in prompt_extractors.items():
                if feat_type == "encodec":
                    feat = extractor.encode(prompt_audio, config["sampling_rate"], device)
                    feat = feat.squeeze(0).cpu().numpy()  # q, t
                else:
                    logging.info(f"Not supported feature type {feat_type}. Skip.")
                    continue
                write_hdf5(
                    feat_path,
                    "prompt_" + feat_type,
                    feat.astype(np.float32),
                )
        new_datas.append({**item, "feat_path": os.path.realpath(feat_path)})

    # write to csv
    write_csv(new_datas, args.csv)


if __name__ == "__main__":
    main()
