#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import re
import soundfile as sf
import pandas as pd
import time
import torch

from joblib import load
from functools import cache
from pathlib import Path
from jatts.trainers.base import Trainer
from jatts.utils import read_hdf5
from encodec import EncodecModel
from encodec.utils import convert_audio
from einops import rearrange

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def flatten_dict(d):
    records = pd.json_normalize(d).to_dict(orient="records")
    return records[0] if records else {}


def _get_named_modules(module, attrname):
    for name, module in module.named_modules():
        if hasattr(module, attrname):
            yield name, module


def gather_attribute(module, attrname, delete=True, prefix=True):
    ret = {}
    for name, module in _get_named_modules(module, attrname):
        ret[name] = getattr(module, attrname)
        if delete:
            try:
                delattr(module, attrname)
            except Exception as e:
                raise RuntimeError(f"{name} {module} {attrname}") from e
    if prefix:
        ret = {attrname: ret}
    ret = flatten_dict(ret)
    # remove consecutive dots
    ret = {re.sub(r"\.+", ".", k): v for k, v in ret.items()}
    return ret


class ValleTrainer(Trainer):
    """Customized trainer module for LM TTS"""

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        xs = [x.to(self.device).long() for x in batch["xs"]]
        prompts = [p.to(self.device).long() for p in batch["pm"]]

        if self.config["model_type"] == "ValleAR":
            # get only the first quantization level as targets
            ys = [y[0, :].to(self.device).long() for y in batch["ys"]]  # t
        elif self.config["model_type"] == "ValleNAR":
            # use all quantization levels as targets
            ys = [y.transpose(1, 0).to(self.device).long() for y in batch["ys"]]  # t, q

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
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        @cache
        def _load_model(device="cuda"):
            # Instantiate a pretrained EnCodec model
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(6.0)
            model.to(device)
            return model

        def unload_model():
            return _load_model.cache_clear()

        @torch.inference_mode()
        def decode(codes, device="cuda"):
            """
            Args:
                codes: (b q t)
            """
            assert codes.dim() == 3
            model = _load_model(device)
            return model.decode([(codes, None)]), model.sample_rate

        def decode_to_file(resps, path: Path):
            assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
            resps = rearrange(resps, "t q -> 1 q t")
            wavs, sr = decode(resps)
            sf.write(str(path), wavs.cpu()[0, 0], sr)

        # parse batch
        xs = [x.to(self.device).long() for x in batch["xs"]]
        ys = [y.to(self.device).long() for y in batch["ys"]]
        prompts = [p.to(self.device).long() for p in batch["pm"]]

        for idx, (x, y, pm) in enumerate(zip(xs, ys, prompts)):
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
                codes = self.model(x, [pm], max_steps=self.config["max_ar_steps"])
                codes = rearrange(codes[0], "t -> 1 1 t")
                wav, sr = decode(codes)
                sf.write(
                    os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                    wav.cpu().numpy()[0, 0],
                    sr,
                    "PCM_16",
                )
            else:
                # NAR mode
                for i in range(1, 8):
                    y_ = [
                        y[:, :i].to(self.device),
                    ]
                    codes = self.model(
                        x,
                        [pm],
                        resps_list=y_,
                        sampling_temperature=0.2,
                    )[0]
                    # codes = ret["outs"]
                    codes = rearrange(codes, "t q -> 1 q t")
                    wav, sr = decode(codes)
                    sf.write(
                        os.path.join(dirname, "wav", f"{idx}_gen_{i}.wav"),
                        wav.cpu().numpy()[0, 0],
                        sr,
                        "PCM_16",
                    )

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(codes.size(0)) / (time.time() - start_time))
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break
