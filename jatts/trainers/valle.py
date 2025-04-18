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
import matplotlib.pyplot as plt


class VALLETrainer(Trainer):
    """Customized trainer module for LM TTS"""

    def _train_step(self, batch):
        """Train model one step."""
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
            ys = [y.transpose(1, 0).to(self.device).long() for y in batch["ys"]]  # q, t -> t, q

        # model forward
        nll_loss = self.model(xs, prompts, ys)

        # loss computation
        gen_loss = 0.0
        nll_loss = self.model.loss

        nll_loss = sum(nll_loss.values())
        self.total_train_loss["train/nll_loss"] += (
            nll_loss.item() / self.gradient_accumulate_steps
        )
        gen_loss += nll_loss

        self.total_train_loss["train/loss"] += (
            gen_loss.item() / self.gradient_accumulate_steps
        )

        # update model
        if self.gradient_accumulate_steps > 1:
            gen_loss = gen_loss / self.gradient_accumulate_steps
        gen_loss.backward()
        self.all_loss += gen_loss.item()
        del gen_loss

        self.backward_steps += 1
        if self.backward_steps % self.gradient_accumulate_steps > 0:
            return

        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.all_loss = 0.0

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # parse batch
        xs = [x.to(self.device).long() for x in batch["xs"]]
        ys = [y.to(self.device).long() for y in batch["ys"]]
        # NOTE(unilight) 20250417: use random training utterance as the prompt during validation
        prompts = [p.to(self.device).long() for p in batch["pm"]]

        for idx, (x, y, pm) in enumerate(zip(xs, ys, prompts)):
            # y: q, t
            start_time = time.time()

            # check directory
            dirname = os.path.join(
                self.config["outdir"], f"predictions/{self.steps}steps"
            )
            if not os.path.exists(os.path.join(dirname, "wav")):
                os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)

            # Check if model is wrapped in DDP
            model = self.model.module if hasattr(self.model, "module") else self.model
            if model.causal:
                # AR mode
                codes = self.model([x], [pm], max_steps=self.config["max_ar_steps"])
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
                    codes = self.model(
                        [x],
                        [pm],
                        resps_list=y_,
                        sampling_temperature=0.2,
                    )[0] # q, t
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
