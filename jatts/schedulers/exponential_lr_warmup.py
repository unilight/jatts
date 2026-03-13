#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import math
from torch.optim.lr_scheduler import ExponentialLR

class WarmupExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, gamma, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.finished_warmup = False
        self.exponential_scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Let ExponentialLR handle it
            if not self.finished_warmup:
                self.finished_warmup = True
                # Reset base_lrs for exponential decay from current lr
                self.exponential_scheduler.base_lrs = [
                    group["lr"] for group in self.optimizer.param_groups
                ]
            return self.exponential_scheduler.get_lr()

    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup_steps:
            self.exponential_scheduler.step()