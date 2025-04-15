from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import math
import os
import time
from collections import defaultdict

# set to avoid matplotlib error in CLI environment
import matplotlib
import soundfile as sf
import torch
from jatts.trainers.base import Trainer
from tensorboardX import SummaryWriter
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_and_save(array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"):
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
        plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
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


class E2TTSTrainer(object):
    """Customized trainer module for E2-TTS training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        vocoder,
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

        # EMA
        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False)
            self.ema_model.to(self.accelerator.device)

        # self.noise_scheduler = noise_scheduler # in the original implementation this is basically none

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
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(
                    self.optimizer
                ).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=self.steps,
            )
            self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Load only model parameters.

        """
        self.accelerator.wait_for_everyone()
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"]
            )
            if not load_only_params:
                self.accelerator.unwrap_model(self.optimizer).load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.steps = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"]
            )
            self.steps = 0

        del checkpoint
        gc.collect()

    def _train_step(self, batch):
        """Train model one step."""
        with self.accelerator.accumulate(self.model):
            # parse batch
            text_inputs = batch["xs"]
            mel_spec = batch["ys"]
            mel_lengths = batch["olens"]

            loss, cond, pred = self.model(
                text=text_inputs,
                feats=mel_spec,
                feats_lengths=mel_lengths,
                # noise_scheduler=self.noise_scheduler, # NOTE(unilight): this is basically none in the original implementation
            )
            self.total_train_loss["train/loss"] += loss.item() / self.gradient_accumulate_steps

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
            if self.is_main:
                self.ema_model.update()

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
    def _eval_step(self, batch):
        """Evaluate model one step."""
        pass

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        # save intermediate result
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # restore mode
        self.model.train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # parse batch
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)

        print("xs", xs.shape)
        print("ys", ys.shape)
        print("ilens", ilens.shape)
        print("olens", olens.shape)

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
            print("x", x.shape, x)
            print(ilen)
            ref_audio_len = olen  # NOTE(unilight): here we just use the output as the reference, which is kind of like cheating
            infer_text = torch.cat((x[:ilen], x[:ilen]), dim=0)

            with torch.inference_mode():
                start_time = time.time()
                outs, _ = self.accelerator.unwrap_model(self.model).inference(
                    cond=y[:olen].unsqueeze(0),
                    text=infer_text.unsqueeze(0),
                    duration=olen * 2,
                    steps=self.config["nfe_step"],
                    cfg_strength=self.config["cfg_strength"],
                    sway_sampling_coef=self.config["sway_sampling_coef"],
                    max_duration=self.config["max_duration"],
                )
                outs = outs[ref_audio_len:].to(torch.float32)
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

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)
            peak_memory = torch.cuda.max_memory_allocated() / (2**30)
            logging.info(f"Peak Memory: {peak_memory:.4f} GB")
            torch.cuda.reset_peak_memory_stats()

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
