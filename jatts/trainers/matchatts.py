#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import math
import os
import time

# set to avoid matplotlib error in CLI environment
import matplotlib
import soundfile as sf
import torch
from jatts.trainers.base import Trainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MatchaTTSTrainer(Trainer):
    """Customized trainer module for Matcha-TTS training."""

    def _train_step(self, batch):
        # parse batch
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)
        if "durations" in batch:
            durations = batch["durations"].to(self.device)
            duration_lens = batch["duration_lens"].to(self.device)
        else:
            durations = None
            duration_lens = None

        spkembs = batch[
            "spkembs"
        ]  # if no spkembs, this is default set to None by collator
        if spkembs is not None:
            spkembs = batch["spkembs"].to(self.device)

        # model forward
        ret = self.model(xs, ilens, ys, olens, durations, duration_lens, spkembs)
        d_outs = ret["d_outs"]

        gen_loss = 0.0

        # cfm loss
        cfm_loss = ret["cfm_loss"]
        self.total_train_loss["train/cfm_loss"] += cfm_loss.item()
        gen_loss += cfm_loss

        # prior loss
        if "EncoderPriorLoss" in self.criterion:
            # ys, hs: [B, T, dim]; h_masks: [B, 1, T]
            encoder_prior_loss = self.criterion["EncoderPriorLoss"](
                ret["hs"], ret["ys"], ret["olens_in"]
            )
        else:
            encoder_prior_loss = 0.0
        self.total_train_loss["train/encoder_prior_loss"] += encoder_prior_loss.item()
        gen_loss += encoder_prior_loss

        # duration loss
        if self.steps > self.config.get("dp_train_start_steps", 0):
            if "DurationPredictorLoss" in self.criterion:
                duration_loss = self.criterion["DurationPredictorLoss"](
                    d_outs, ret["ds"], ilens
                )
                self.total_train_loss["train/duration_loss"] += duration_loss.item()

        else:
            duration_loss = 0.0
            self.total_train_loss["train/duration_loss"] += 0.0
        gen_loss += duration_loss

        # forward sum loss
        if self.steps < self.config.get("dp_train_start_steps", 0):
            if "ForwardSumLoss" in self.criterion:
                log_p_attn = ret["log_p_attn"]
                forwardsum_loss = self.criterion["ForwardSumLoss"](
                    log_p_attn, ilens, olens
                )
                self.total_train_loss[
                    "train/forward_sum_loss"
                ] += forwardsum_loss.item()
        else:
            forwardsum_loss = 0.0
            self.total_train_loss["train/forward_sum_loss"] += 0.0

        gen_loss += self.config["lambda_align"] * forwardsum_loss

        # bin loss
        if self.steps > self.config.get("bin_loss_start_steps", 0):
            bin_loss = ret["bin_loss"]
            self.total_train_loss["train/binary_loss"] += bin_loss.item()
        else:
            bin_loss = 0.0
            self.total_train_loss["train/binary_loss"] += 0.0

        gen_loss += self.config["lambda_align"] * bin_loss

        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(
            array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
        ):
            shape = array.shape
            if len(shape) == 1:
                # for eos probability
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(array)
                plt.xlabel("Frame")
                plt.ylabel("Probability")
                plt.ylim([0, 1])
            elif len(shape) == 2:
                # for tacotron 2 attention weights, whose shape is (out_length, in_length)
                if ref is None:
                    plt.figure(figsize=figsize, dpi=dpi)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                else:
                    plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                    plt.subplot(1, 2, 1)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                    plt.subplot(1, 2, 2)
                    plt.imshow(ref.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
            elif len(shape) == 4:
                # for transformer attention weights,
                # whose shape is (#leyers, #heads, out_length, in_length)
                plt.figure(
                    figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
                )
                for idx1, xs in enumerate(array):
                    for idx2, x in enumerate(xs, 1):
                        plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                        plt.imshow(x, aspect="auto")
                        plt.xlabel("Input")
                        plt.ylabel("Output")
            else:
                raise NotImplementedError("Support only from 1D to 4D array.")
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(figname)):
                # NOTE: exist_ok = True is needed for parallel process decoding
                os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname)
            plt.close()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # parse batch
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)

        spkembs = batch[
            "spkembs"
        ]  # if no spkembs, this is default set to None by collator
        if spkembs is not None:
            spkembs = batch["spkembs"].to(self.device)
        else:
            spkembs = [
                None for _ in range(xs.shape[0])
            ]  # this is because we need to iterate over spkembs

        # generate
        for idx, (x, y, ilen, olen, spkemb) in enumerate(
            zip(xs, ys, ilens, olens, spkembs)
        ):
            start_time = time.time()
            if self.config["distributed"]:
                ret = self.model.module.inference(
                    x[:ilen],
                    y[:olen],
                    spembs=spkemb,
                    temperature=self.config["temperature"],
                    n_timesteps=self.config["ode_steps"],
                )
            else:
                ret = self.model.inference(
                    x[:ilen],
                    y[:olen],
                    spembs=spkemb,
                    temperature=self.config["temperature"],
                    n_timesteps=self.config["ode_steps"],
                )

            outs = ret["feat_gen"]
            d_outs = ret["duration"]
            ds = ret["ds"]

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )
            logging.info(
                "duration from alignment module:   {}".format(
                    " ".join([str(int(d)) for d in ds.cpu().numpy()])
                )
            )
            logging.info(
                "duration from duration predictor: {}".format(
                    " ".join([str(int(d)) for d in d_outs.cpu().numpy()])
                )
            )

            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y[:olen].cpu().numpy(),
                origin="lower",
            )

            if "log_p_attn" in ret:
                log_p_attn = ret["log_p_attn"]
                _plot_and_save(
                    log_p_attn.cpu().numpy(),
                    dirname + f"/alignment/{idx}.png",
                    origin="lower",
                )

            if self.vocoder is not None:
                if not os.path.exists(os.path.join(dirname, "wav")):
                    os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)
                y, sr = self.vocoder.decode(outs)
                sf.write(
                    os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                    y.cpu().numpy(),
                    sr,
                    "PCM_16",
                )

            # write duration
            if not os.path.exists(os.path.join(dirname, "durations")):
                os.makedirs(os.path.join(dirname, "durations"), exist_ok=True)
            d_outs = [str(d) for d in d_outs.cpu().numpy().tolist()]
            if "durations" in batch:
                d_gts = [
                    str(d) for d in batch["durations"][idx][:ilen].numpy().tolist()
                ]
            else:
                d_gts = [str(int(d)) for d in ds.cpu().numpy()]
            with open(dirname + f"/durations/{idx}.txt", "w") as f:
                for d_gt, d_out in zip(d_gts, d_outs):
                    f.write(f"{d_gt} {d_out}\n")

            if idx >= self.config["num_save_intermediate_results"]:
                break
