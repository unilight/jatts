#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Flow-matching related modules.
Source: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
"""

from abc import ABC

import torch
import torch.nn.functional as F
from jatts.modules.matchatts.decoder import Decoder


class CFM(torch.nn.Module, ABC):
    def __init__(
        self,
        in_channels,
        out_channel,
        channels=[256, 256],
        dropout: float = 0.05,
        attention_head_dim: int = 64,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
        num_heads: int = 2,
        act_fn: str = "snakebeta",
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.sigma_min = sigma_min

        self.estimator = Decoder(
            in_channels=in_channels,
            out_channels=out_channel,
            channels=channels,
            dropout=dropout,
            attention_head_dim=attention_head_dim,
            n_blocks=n_blocks,
            num_mid_blocks=num_mid_blocks,
            num_heads=num_heads,
            act_fn=act_fn,
        )

    @torch.inference_mode()
    def inference(self, mu, mask, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask)

    def solve_euler(self, x, t_span, mu, mask):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def forward(self, mu):
        """Forward (for training)

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze()), u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])
        return loss, y

    def compute_loss(self, x1, mask, mu):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze()), u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])
        return loss, y
