#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train an E2-TTS model."""

import argparse
import logging
import os
import sys

import jatts
import jatts.collaters
import jatts.losses
import jatts.models
import jatts.trainers
import numpy as np
import torch
import yaml
from jatts.datasets.tts_dataset import TTSDataset, DynamicBatchSampler
from jatts.schedulers.warmup_lr import WarmupLR
from jatts.schedulers.e2tts_scheduler import E2TTSSequentialLR

from jatts.utils import read_hdf5
from jatts.vocoder import Vocoder
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader

# from jatts.vocoder.s3prl_feat2wav import S3PRL_Feat2Wav
# from jatts.vocoder.griffin_lim import Spectrogram2Waveform
# from jatts.vocoder.encodec import EnCodec_decoder


scheduler_classes = {
    "warmuplr": WarmupLR,
    "exponentiallr": ExponentialLR,
    "StepLR": StepLR,
    "E2TTSSequentialLR": E2TTSSequentialLR,
}


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train an E2-TTS model (See detail in bin/e2tts_train.py).")
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        type=str,
        help=("training csv file. "),
    )
    parser.add_argument(
        "--dev-csv",
        required=True,
        type=str,
        help=("development csv file. "),
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
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = jatts.__version__  # add version info

    out_feat_type = config["out_feat_type"]

    # write idim
    with open(args.token_list, encoding="utf-8") as f:
        token_list = [line.rstrip() for line in f]
    vocab_size = len(token_list)
    logging.info(f"Vocabulary size: {vocab_size }")
    config["model_params"]["idim"] = vocab_size

    # save config
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    train_dataset = TTSDataset(
        csv_path=args.train_csv,
        stats_path=args.stats,
        feat_list=config["feat_list"],
        token_list_path=args.token_list,
        token_column=args.token_column,
        sampling_rate=config["sampling_rate"],
        hop_size=config["hop_size"],
        is_inference=False,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = TTSDataset(
        csv_path=args.dev_csv,
        stats_path=args.stats,
        feat_list=config["feat_list"],
        token_list_path=args.token_list,
        token_column=args.token_column,
        sampling_rate=config["sampling_rate"],
        hop_size=config["hop_size"],
        is_inference=False,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater_class = getattr(
        jatts.collaters,
        config.get("collater_type", "ARTTSCollater"),
    )
    collater = collater_class()
    sampler = {
        "train": DynamicBatchSampler(
            dataset["train"],
            config["batch_size_per_gpu"],
            max_samples=config["max_samples"],
            random_seed=config["sampler_random_seed"],  # This enables reproducible shuffling
            drop_residual=False,
        ),
        "dev": None,
    }
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            collate_fn=collater,
            num_workers=config["num_workers"],
            batch_sampler=sampler["train"],
            pin_memory=config["pin_memory"],
            persistent_workers=True,
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            collate_fn=collater,
            batch_size=config["num_save_intermediate_results"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    model_class = getattr(
        jatts.models,
        config.get("model_type", "TransformerTTS"),
    )
    model = model_class(
        **config["model_params"],
    )

    # load output stats for denormalization
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
            stats=stats,
            n_fft=config["fft_size"],
            n_shift=config["hop_size"],
            fs=config["sampling_rate"],
            n_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            griffin_lim_iters=64,
        )

    # define criterions -> no need to specify for E2TTS? (only a simple MSE loss, calculated in model file)

    # define optimizers and schedulers
    optimizer_class = getattr(
        torch.optim,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        model.parameters(),
        **config["optimizer_params"],
    )
    scheduler_class = scheduler_classes.get(config.get("scheduler_type", "warmuplr"))
    scheduler = scheduler_class(
        optimizer=optimizer,
        **config["scheduler_params"],
    )

    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        model = DistributedDataParallel(model)

    # show settings
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)

    # define trainer
    trainer_class = getattr(
        jatts.trainers,
        config.get("trainer_type", "ARTTSTrainer"),
    )
    trainer = trainer_class(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        vocoder=vocoder,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
