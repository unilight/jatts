#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""Train TTS model."""

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
from jatts.datasets.tts_dataset import TTSDataset
from jatts.schedulers.warmup_lr import WarmupLR

# from jatts.losses import Seq2SeqLoss, GuidedMultiHeadAttentionLoss
from jatts.utils import read_hdf5
from jatts.vocoder import Vocoder
from jatts.modules.feature_extract.encodec import EnCodec
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader

# from jatts.vocoder.s3prl_feat2wav import S3PRL_Feat2Wav
# from jatts.vocoder.griffin_lim import Spectrogram2Waveform
# from jatts.vocoder.encodec import EnCodec_decoder


scheduler_classes = {
    "warmuplr": WarmupLR,
    "exponentiallr": ExponentialLR,
    "StepLR": StepLR,
}


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train TTS model (See detail in bin/tts_train.py).")
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
        required=False,
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
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of processes for distributed training. No need to explicitly specify.",
    )
    args = parser.parse_args()

    # Initialize distributed training settings
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1

    args.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
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

    logging.info(f"World size: {args.world_size}")
    logging.info(f"Local rank: {args.rank}")

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.rank = torch.distributed.get_rank()

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.rank}")
        torch.cuda.set_device(args.rank)
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

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

    if not args.stats:
        stats = None

    # get dataset
    train_dataset = TTSDataset(
        csv_path=args.train_csv,
        stats_path=args.stats,
        feat_list=config["feat_list"],
        prompt_feat_list=config.get("prompt_feat_list", None),
        token_list_path=args.token_list,
        token_column=args.token_column,
        is_inference=False,
        prompt_strategy=config.get("prompt_strategy", "same"),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = TTSDataset(
        csv_path=args.dev_csv,
        stats_path=args.stats,
        feat_list=config["feat_list"],
        prompt_feat_list=config.get("prompt_feat_list", None),
        token_list_path=args.token_list,
        token_column=args.token_column,
        is_inference=False,
        prompt_strategy=config.get("prompt_strategy", "same"),
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
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collater,
            batch_size=config["batch_size"],
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
    ).to(device)

    logging.info(f"stats: {args.stats}")
    # load output stats for denormalization
    if args.stats is not None:
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
        elif vocoder_type in ["encodec", "encodec_24khz"]:
            vocoder = EnCodec(fs=24, device=device)
        elif vocoder_type == "encodec_48khz":
            vocoder = EnCodec(fs=48, device=device)
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

    # define criterions
    if config.get("criterions", None) is not None:
        criterion = {
            criterion_class: getattr(jatts.losses, criterion_class)(
                **criterion_paramaters
            )
            for criterion_class, criterion_paramaters in config["criterions"].items()
        }
    else:
        logging.info("No criterions are specified. Please make sure this is intended.")
        criterion = None

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
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=True,
        )
        logging.info("Model wrapped with DistributedDataParallel.")

    # show settings
    logging.info(model)
    logging.info(optimizer)
    logging.info(scheduler)
    if criterion is not None:
        logging.info(criterion)

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
        criterion=criterion,
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
