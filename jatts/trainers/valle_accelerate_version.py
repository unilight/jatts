#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import re
# import pandas as pd
import time
from functools import cache
from pathlib import Path

# set to avoid matplotlib error in CLI environment
import matplotlib
import soundfile as sf
import torch
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from jatts.trainers.base import Trainer
from jatts.utils import read_hdf5
from joblib import load

matplotlib.use("Agg")
import gc
from collections import defaultdict

import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tensorboardX import SummaryWriter
from tqdm import tqdm


class VALLETrainer(Trainer):
    """Customized trainer module for LM TTS"""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        vocoder,
        criterion,  # dummy in VALL-E
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.vocoder = vocoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.gradient_accumulate_steps = self.config.get("gradient_accumulate_steps", 1)
        self.device = device

        # accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.gradient_accumulate_steps,
        )
        self.model, self.optimizer, self.data_loader["train"], self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.data_loader["train"], self.scheduler
            )
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def run(self):
        """Run training."""
        self.model.train()
        self.backward_steps = 0
        self.all_loss = 0.0
        self.tqdm = tqdm(
            initial=self.steps,
            total=self.config["train_max_steps"],
            desc="[train]",
            disable=not self.accelerator.is_local_main_process,
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        self.accelerator.end_training()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler=self.scheduler.state_dict(),
                steps=self.steps,
                epochs=self.epochs,
            )
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Load only model parameters.

        """
        self.accelerator.wait_for_everyone()
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        if not load_only_params:
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(
                checkpoint["optimizer"]
            )
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]

        del state_dict
        gc.collect()

    def _train_step(self, batch):
        """Train model one step."""
        with self.accelerator.accumulate(self.model):
            # parse batch
            xs = [x.to(self.device).long() for x in batch["xs"]]
            prompts = [p.to(self.device).long() for p in batch["pm"]]

            # NOTE(unilight) 20250417: use the same utterance as the prompt during training
            # remember to transpose!
            # prompts = [p.transpose(1, 0).to(self.device).long() for p in batch["ys"]] # q, t -> t, q

            if self.config["model_type"] == "VALLEAR":
                # get only the first quantization level as targets
                ys = [y[0, :].to(self.device).long() for y in batch["ys"]]  # t
            elif self.config["model_type"] == "VALLENAR":
                # use all quantization levels as targets
                ys = [
                    y.transpose(1, 0).to(self.device).long() for y in batch["ys"]
                ]  # q, t -> t, q

            # model forward
            nll_loss = self.model(xs, prompts, ys)

            # loss computation
            nll_loss = self.accelerator.unwrap_model(self.model).loss
            nll_loss = sum(nll_loss.values())
            self.total_train_loss["train/nll_loss"] += (
                nll_loss.item() / self.gradient_accumulate_steps
            )
            loss = nll_loss

            self.total_train_loss["train/loss"] += (
                loss.item() / self.gradient_accumulate_steps
            )

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.config["grad_norm"] > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config["grad_norm"]
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            # update counts
            self.steps += 1
            self.tqdm.update(1)

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()
                self._check_train_finish()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # shuffle sampler
        self.data_loader["train"].batch_sampler.set_epoch(self.epochs)

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # parse batch
        xs = [x.to(self.device).long() for x in batch["xs"]]
        ys = [y.to(self.device).long() for y in batch["ys"]]
        # NOTE(unilight) 20250417: use random training utterance as the prompt during validation
        prompts = [p.to(self.device).long() for p in batch["pm"]]

        for idx, (x, y, pm) in enumerate(zip(xs, ys, prompts)):
            # y: q, t
            with torch.inference_mode():
                start_time = time.time()

                if not os.path.exists(os.path.join(dirname, "wav")):
                    os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)

                # Check if model is wrapped in DDP
                if self.accelerator.unwrap_model(self.model).causal:
                    # AR mode
                    codes = self.accelerator.unwrap_model(self.model)(
                        [x], [pm], max_steps=self.config["max_ar_steps"]
                    )
                    codes = rearrange(codes[0], "t -> 1 1 t")
                    assert codes.dim() == 3
                    wav = self.vocoder.decode([(codes, None)])
                    sf.write(
                        os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                        wav.cpu().numpy()[0, 0],
                        self.vocoder.sample_rate,
                        "PCM_16",
                    )
                else:
                    # NAR mode
                    for i in range(1, 8):
                        y_ = [
                            y[:i].to(self.device),
                        ]
                        codes = self.accelerator.unwrap_model(self.model)(
                            [x],
                            [pm],
                            resps_list=y_,
                            sampling_temperature=0.2,
                        )[
                            0
                        ]  # q, t
                        codes = rearrange(codes, "q t -> 1 q t")
                        assert codes.dim() == 3
                        wav = self.vocoder.decode([(codes, None)])
                        sf.write(
                            os.path.join(dirname, "wav", f"{idx}_gen_{i}.wav"),
                            wav.cpu().numpy()[0, 0],
                            self.vocoder.sample_rate,
                            "PCM_16",
                        )

                logging.info(
                    "inference speed = generated 1 second of waveform takes %.1f seconds."
                    % (
                        int(wav.shape[2] / self.vocoder.sample_rate)
                        / (time.time() - start_time)
                    )
                )

            # save prompt
            wav = self.vocoder.decode([(rearrange(pm, "t q -> 1 q t"), None)])
            sf.write(
                os.path.join(dirname, "wav", f"{idx}_prompt.wav"),
                wav.cpu().numpy()[0, 0],
                self.vocoder.sample_rate,
                "PCM_16",
            )

            # save gt
            wav = self.vocoder.decode([(rearrange(y, "q t -> 1 q t"), None)])
            sf.write(
                os.path.join(dirname, "wav", f"{idx}_gt.wav"),
                wav.cpu().numpy()[0, 0],
                self.vocoder.sample_rate,
                "PCM_16",
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break
