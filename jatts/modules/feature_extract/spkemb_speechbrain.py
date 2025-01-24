#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Speaker embedding extractor based on SpeechBrain."""

import numpy as np
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


class SpeechBrainSpkEmbExtractor:
    def __init__(self, device="cpu"):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
        )

    def forward(
        self,
        wav_path: str,
    ) -> np.array:

        signal, fs = torchaudio.load(wav_path)
        embeddings = self.classifier.encode_batch(signal)

        return embeddings.cpu().numpy().reshape(-1)
