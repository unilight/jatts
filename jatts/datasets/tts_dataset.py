# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import torch
from jatts.utils import read_csv, read_hdf5
from jatts.utils.token_id_converter import TokenIDConverter
# from jatts.utils.prompt import prepare_prompt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler, SequentialSampler
from tqdm import tqdm


class TTSDataset(Dataset):
    """PyTorch compatible dataset for TTS."""

    def __init__(
        self,
        csv_path,
        stats_path,
        feat_list,
        token_list_path,
        token_column,
        is_inference,
        prompt_feat_list=None,
        prompt_strategy="same",
        sampling_rate=None,
        hop_size=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            csv_path (str): path to the csv file
            stats_path (str): path to the stats file
            feat_list (list): list of feature names
            token_list_path (str): path to token list
            token_column (str): which column to use as the token in the csv file
            is_inference (bool): if True, do not load features.
            prompt_feat_list (list): list of prompt feature names
            prompt_strategy (str): strategy for prompt features: "same", or "given"
            sampling_rate (int): sampling rate of the audio
            hop_size (int): hop size of the audio
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """

        self.feat_list = feat_list
        self.token_column = token_column
        self.is_inference = is_inference
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.prompt_feat_list = prompt_feat_list
        self.prompt_strategy = prompt_strategy

        # read dataset
        self.dataset, _ = read_csv(csv_path, dict_reader=True)

        # read stats (in training)
        if not self.is_inference:
            self.stats = {}
            for feat_name in feat_list:
                if feat_name in ["encodec", "encodec_24khz", "encodec_48khz"]:
                    continue
                scaler = StandardScaler()
                scaler.mean_ = read_hdf5(stats_path, f"{feat_name}_mean")
                scaler.scale_ = read_hdf5(stats_path, f"{feat_name}_scale")
                self.stats[feat_name] = scaler

        self.token_id_converter = TokenIDConverter(
            token_list=token_list_path,
            unk_symbol="<unk>",
        )

        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.dataset))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: dictionary of items to return

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        item = self.dataset[idx]

        # load and process text
        text = item[self.token_column]
        if self.token_column == "phonemes":
            tokens = [p for p in text.split(" ") if p != ""]
        token_indices = np.array(
            self.token_id_converter.tokens2ids(tokens), dtype=np.int64
        )
        item["tokens"] = tokens
        item["token_indices"] = token_indices

        # load durations (in training)
        if not self.is_inference and "durations" in item:
            item["durations_int"] = np.array(
                [int(d) for d in item["durations"].split(" ")]
            )

        # load features (in training)
        if not self.is_inference:
            for feat_name in self.feat_list:
                raw_feat = read_hdf5(item["feat_path"], feat_name)

                # spkemb: [dim] -> [1, dim]
                if feat_name == "spkemb":
                    raw_feat = raw_feat.reshape(1, -1)

                # pitch, energy: [n_frames] -> [n_frames, 1]
                elif feat_name in ["pitch", "energy"]:
                    raw_feat = raw_feat.reshape(-1, 1)

                if feat_name in ["encodec", "encodec_24khz", "encodec_48khz"]:
                    normalized_feat = raw_feat
                else:
                    normalized_feat = self.stats[feat_name].transform(raw_feat)

                if feat_name == "spkemb":
                    normalized_feat = np.squeeze(normalized_feat, 0)

                item[feat_name] = normalized_feat

        # load prompt
        if self.prompt_strategy == "given":
            assert "prompt_wav_path" in item, "prompt_wav_path must be given if prompt_strategy is 'given'."

            item["prompt_wav_path"] = item["prompt_wav_path"]
            item["prompt_start"] = item["prompt_start"]
            item["prompt_end"] = item["prompt_end"]
            if "prompt_phonemes" in item:
                prompt_phonemes = [p for p in item["prompt_phonemes"].split(" ") if p != ""]
                prompt_indices = np.array(
                    self.token_id_converter.tokens2ids(prompt_phonemes), dtype=np.int64
                )
                item["prompt_phonemes"] = prompt_phonemes
                item["prompt_indices"] = prompt_indices

            # load audio codec features (in training)
            if not self.is_inference:
                for feat_name in self.prompt_feat_list:
                    raw_feat = read_hdf5(item["feat_path"], "prompt_" + feat_name)

                    if feat_name in ["encodec", "encodec_24khz", "encodec_48khz"]:
                        # [n_RVQ, n_frames] -> [n_frames, n_RVQ]
                        raw_feat = raw_feat.transpose(1, 0)

                    item["prompt_" + feat_name] = raw_feat

        elif self.prompt_strategy == "same":
            assert not self.is_inference, "prompt_strategy 'same' is only available in training."

            # load audio codec features from same utterance
            for feat_name in self.prompt_feat_list:
                raw_feat = read_hdf5(item["feat_path"], feat_name)

                if feat_name in ["encodec", "encodec_24khz", "encodec_48khz"]:
                    # [n_RVQ, n_frames] -> [n_frames, n_RVQ]
                    raw_feat = raw_feat.transpose(1, 0)

                item["prompt_" + feat_name] = raw_feat

        if self.allow_cache:
            self.caches[idx] = item

        return item

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.dataset)

    def get_frame_len(self, index):
        """Get the number of frames for the specified index.
        Args:
            index (int): Index of the item.
        Returns:
            int: The number of frames.
        """
        item = self.dataset[index]
        return (
            (float(item["end"]) - float(item["start"]))
            * self.sampling_rate
            / self.hop_size
        )


class DynamicBatchSampler(Sampler[list[int]]):
    """
    This module is for E2-TTS training.
    
    Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self,
        dataset,
        frames_threshold: int,
        max_samples=0,
        random_seed=None,
        drop_residual: bool = False,
    ):

        self.sampler = SequentialSampler(dataset)
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler,
            desc="Sorting with sampler... if slow, check whether dataset is provided with duration",
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices,
            desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu",
        ):
            if batch_frames + frame_len <= self.frames_threshold and (
                max_samples == 0 or len(batch) < max_samples
            ):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)
