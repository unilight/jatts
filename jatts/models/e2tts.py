from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E2-TTS, based on the implementation in https://github.com/SWivid/F5-TTS."""

import logging
from random import random

import torch
import torch.nn.functional as F
from jatts.modules.e2tts.unett import UNetT
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from typeguard import typechecked


def lens_to_mask(
    t: int["b"], length: int | None = None
) -> bool["b n"]:  # noqa: F722 F821
    if length is None:
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(
    seq_len: int["b"], start: int["b"], end: int["b"]
):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(
    seq_len: int["b"], frac_lengths: float["b"]
):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


class E2TTS(torch.nn.Module):
    """E2-TTS module.

    This is an implementation E2-TTS described in `E2 TTS: Embarrassingly
    Easy Fully Non-Autoregressive Zero-Shot TTS`_.

    .. _`E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS`:
        https://arxiv.org/abs/2406.18009

    """

    @typechecked
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        backbone: "str",
        dim: int = 1024,
        depth: int = 24,
        heads: int = 16,
        ff_mult: int = 4,
        text_mask_padding: bool = False,
        pe_attn_head: int = 1,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ):
        # initialize base classes
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.dim = dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        if backbone == "UNetT":
            self.backbone = UNetT(
                text_num_embeds=idim,
                mel_dim=odim,
                dim=dim,
                depth=depth,
                heads=heads,
                ff_mult=ff_mult,
                text_mask_padding=text_mask_padding,
                pe_attn_head=pe_attn_head,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        text: torch.Tensor,
        # text_lengths: torch.Tensor, # we don't need this because it is padded to feats_length
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ):

        batch, seq_len, dtype, device, sigma = (
            *feats.shape[:2],
            feats.dtype,
            self.device,
            self.sigma,
        )

        # get a random span to mask out for training conditionally
        frac_lengths = (
            torch.zeros((batch,), device=self.device)
            .float()
            .uniform_(*self.frac_lengths_mask)
        )
        rand_span_mask = mask_from_frac_lengths(feats_lengths, frac_lengths)

        # x1 is the mel spectrogram
        # x0 is gaussian noise
        x1 = feats
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.backbone(
            x=phi,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    def inference(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: torch.Tensor,
        duration: int | int["b"],  # noqa: F821
        ref_lens: int["b"] | None = None,  # noqa: F821
        # ODE related
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        max_duration=3000,
        seed=None,
        duplicate_test=False,  # NOTE(unilight): what is this for?
        edit_mask=None,
        no_ref_audio=False,
        t_inter=0.1,
    ):

        cond = cond.to(self.device)

        batch_size, cond_seq_len, device = *cond.shape[:2], cond.device
        if ref_lens is None:
            ref_lens = torch.full(
                (batch_size,), cond_seq_len, device=device, dtype=torch.long
            )

        cond_mask = lens_to_mask(ref_lens)

        # NOTE(unilight): editting: future work
        # if edit_mask is not None:
        #     cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full(
                (batch_size,), duration, device=device, dtype=torch.long
            )

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), ref_lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0
            )

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch_size > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.backbone(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
                cache=True,
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.backbone(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
                cache=True,
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if seed is not None:
                torch.manual_seed(seed)
            y0.append(
                torch.randn(dur, self.odim, device=self.device, dtype=step_cond.dtype)
            )
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(
            t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype
        )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.backbone.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        return out.squeeze(0), trajectory
