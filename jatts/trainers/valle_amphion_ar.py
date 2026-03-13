#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# This file is adapted from: https://github.com/open-mmlab/Amphion
# Copyright 2023 Amphion team, licensed under MIT

import logging
import os
import re
import numpy as np

import time
from functools import cache
from pathlib import Path

# set to avoid matplotlib error in CLI environment
import matplotlib
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio, save_audio
from jatts.trainers.base import Trainer
from jatts.utils import read_hdf5
from joblib import load

from jatts.modules.utils import make_non_pad_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_FRAMES = 450 # if the generated waveform is shorter than this, we will not save it


class VALLE_AMPHION_ARTrainer(Trainer):
    """Customized trainer module for training VALL-E AR model, implemented by Amphion."""

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        xs = [x.to(self.device).long() for x in batch["xs"]]
        prompts = [p.to(self.device).long() for p in batch["pm"]]
        # get only the first quantization level as targets
        ys = [y[0, :].to(self.device).long() for y in batch["ys"]]  # t

        x_lens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        prompt_lens = torch.from_numpy(np.array([prompt.shape[0] for prompt in prompts])).long() # CHECK WHETHER WE NEED TO FLATTEN THIS
        y_lens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()

        xs = pad_sequence(xs, batch_first=True, padding_value=0)
        prompts = pad_sequence(prompts, batch_first=True, padding_value=0)
        ys = pad_sequence(ys, batch_first=True, padding_value=0)

        x_mask = make_non_pad_mask(x_lens).to(torch.long).to(self.device)
        y_mask = make_non_pad_mask(y_lens).to(torch.long).to(self.device)

        # model forward
        out = self.model(
            phone_ids=xs,
            phone_mask=x_mask,
            target_ids=ys,
            target_mask=y_mask,
        )
        loss = out.loss

        # loss computation
        self.total_train_loss["train/nll_loss"] += (
            loss.item() / self.gradient_accumulate_steps
        )

        self.total_train_loss["train/loss"] += (
            loss.item() / self.gradient_accumulate_steps
        )

        # update model
        if self.gradient_accumulate_steps > 1:
            loss = loss / self.gradient_accumulate_steps
        loss.backward()
        self.all_loss += loss.item()

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
        ys = [y.transpose(1, 0).to(self.device).long() for y in batch["ys"]]
        # get only the first quantization level as targets
        prompts = [p.to(self.device).long() for p in batch["pm"]] # each element: [t, q]
        
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

            codes = model.sample_hf(
                x.unsqueeze(0),
                pm.transpose(1, 0)[0].unsqueeze(0),
                temperature=0.9
            ).squeeze(0) # [t]
            codes = rearrange(codes, "t -> 1 1 t")
            assert codes.dim() == 3
            if codes.shape[2] < MIN_FRAMES:
                logging.info(
                    f"Generated waveform is too short ({codes.shape[2]} frames), skipping saving."
                )
                continue
            if torch.max(codes).item() >= self.config["model_params"]["target_vocab_size"]:
                logging.info(
                    f"Generated waveform has out-of-range values ({torch.max(codes).item()}), skipping saving."
                )
                continue
            wav = (
                self.vocoder.model.decode([(codes, None)]).squeeze(0).cpu()
            )  # 1/2, t
            save_audio(
                wav,
                os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                self.vocoder.model.sample_rate,
                rescale=self.vocoder.rescale,
            )


            logging.info(
                "inference speed = generated 1 second of waveform takes %.2f seconds."
                % (
                    int(wav.shape[1] / self.vocoder.model.sample_rate)
                    / (time.time() - start_time)
                )
            )

            # save prompt
            wav = (
                self.vocoder.model.decode([(rearrange(pm, "t q -> 1 q t"), None)])
                .squeeze(0)
                .cpu()
            )  # 1/2, t
            save_audio(
                wav,
                os.path.join(dirname, "wav", f"{idx}_prompt.wav"),
                self.vocoder.model.sample_rate,
                rescale=self.vocoder.rescale,
            )

            # save gt
            wav = (
                self.vocoder.model.decode([(rearrange(y, "t q -> 1 q t"), None)])
                .squeeze(0)
                .cpu()
            )  # 1/2, t
            save_audio(
                wav,
                os.path.join(dirname, "wav", f"{idx}_gt.wav"),
                self.vocoder.model.sample_rate,
                rescale=self.vocoder.rescale,
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break
