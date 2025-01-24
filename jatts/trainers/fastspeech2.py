#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import soundfile as sf
import time
import torch

from jatts.trainers.base import Trainer

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FastSpeech2Trainer(Trainer):
    """Customized trainer module for FastSpeech2 training."""

    def _train_step(self, batch):
        # parse batch
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)
        pitches = batch["pitch"].to(self.device)
        pitch_lengths = batch["pitch_lens"].to(self.device)
        energys = batch["energys"].to(self.device)
        energy_lengths = batch["energy_lens"].to(self.device)
        durations = batch["durations"].to(self.device)
        duration_lens = batch["duration_lens"].to(self.device)

        spkembs = batch["spkembs"] # if no spkembs, this is default set to None by collator
        if spkembs is not None:
            spkembs = batch["spkembs"].to(self.device)

        # model forward
        ret = self.model(xs, ilens, ys, olens, durations, duration_lens, pitches, pitch_lengths, energys, energy_lengths, spkembs)
        after_outs = ret["after_outs"]
        before_outs = ret["before_outs"]
        d_outs = ret["d_outs"]
        p_outs = ret["p_outs"]
        e_outs = ret["e_outs"]
        ys_ = ret["ys"]
        olens_ = ret["olens"]

        # mel loss (default L1 loss)
        mel_loss = self.criterion["MelLoss"](after_outs, before_outs, ys_, olens_)
        self.total_train_loss["train/mel_loss"] += mel_loss.item()

        # duration loss
        duration_loss = self.criterion["DurationPredictorLoss"](d_outs, durations, ilens)
        self.total_train_loss["train/duration_loss"] += duration_loss.item()

        # pitch loss
        pitch_loss = self.criterion["PitchLoss"](p_outs, pitches, ilens)
        self.total_train_loss["train/pitch_loss"] += pitch_loss.item()

        # energy loss
        energy_loss = self.criterion["EnergyLoss"](e_outs, energys, ilens)
        self.total_train_loss["train/energy_loss"] += energy_loss.item()

        gen_loss = mel_loss + duration_loss + pitch_loss + energy_loss
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

        spkembs = batch["spkembs"] # if no spkembs, this is default set to None by collator
        if spkembs is not None:
            spkembs = batch["spkembs"].to(self.device)
        else:
            spkembs = [None for _ in range(xs.shape[0])] # this is because we need to iterate over spkembs

        # generate
        for idx, (x, y, ilen, olen, spkemb) in enumerate(zip(xs, ys, ilens, olens, spkembs)):
            start_time = time.time()
            if self.config["distributed"]:
                ret = self.model.module.inference(
                    x[:ilen], spembs=spkemb
                )
            else:
                ret = self.model.inference(
                    x[:ilen], spembs=spkemb
                )
            
            outs = ret["feat_gen"]
            d_outs = ret["duration"]

            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y[:olen].cpu().numpy(),
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
            d_gts = [str(d) for d in batch["durations"][idx][:ilen].numpy().tolist()]
            with open(dirname + f"/durations/{idx}.txt", "w") as f:
                for d_gt, d_out in zip(d_gts, d_outs):
                    f.write(f"{d_gt} {d_out}\n")

            if idx >= self.config["num_save_intermediate_results"]:
                break
