# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os
from multiprocessing import Manager
from pathlib import Path

import numpy as np
from jatts.utils import read_csv, read_hdf5
from jatts.utils.token_id_converter import TokenIDConverter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


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
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            csv_path (str): path to the csv file
            stats_path (str): path to the stats file
            feat_list (list): list of feature names
            token_list (list): path to token list
            token_column (str): which column to use as the token in the csv file
            is_inference (bool): if True, do not load features.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """

        self.feat_list = feat_list
        self.token_column = token_column
        self.is_inference = is_inference

        # read dataset
        self.dataset, _ = read_csv(csv_path, dict_reader=True)

        # read stats (in training)
        if not self.is_inference:
            self.stats = {}
            for feat_name in feat_list:
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
        if not self.is_inference:
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
                normalized_feat = self.stats[feat_name].transform(raw_feat)

                if feat_name == "spkemb":
                    normalized_feat = np.squeeze(normalized_feat, 0)

                item[feat_name] = normalized_feat

        if self.allow_cache:
            self.caches[idx] = item

        return item

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.dataset)
