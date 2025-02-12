#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import h5py
import numpy as np
import yaml
from jatts.utils import read_csv, read_hdf5, write_hdf5
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in bin/compute_statistics.py)."
        )
    )
    parser.add_argument("--csv", required=True, help="csv file path")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help=(
            "path to save statistics. if not provided, "
            "stats will be saved in the above root directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # get dataset
    dataset, _ = read_csv(args.csv, dict_reader=True)
    logging.info(f"The number of files = {len(dataset)}.")

    # get all feature names
    with h5py.File(dataset[0]["feat_path"], "r") as hdf_file:
        group = hdf_file["/"]  # Access the specified group
        feat_names = [name for name in group.keys()]

    # calculate statistics
    for feat_name in feat_names:
        if feat_name == "wave":
            continue

        logging.info(f"Calculating statistics for {feat_name}")

        scaler = StandardScaler()
        for line in tqdm(dataset):
            feat = read_hdf5(line["feat_path"], feat_name)

            # spkemb: [dim] -> [1, dim]
            if feat_name == "spkemb":
                feat = feat.reshape(1, -1)

            # pitch, energy: [n_frames] -> [n_frames, 1]
            elif feat_name in ["pitch", "energy"]:
                feat = feat.reshape(-1, 1)
            scaler.partial_fit(feat)

        write_hdf5(
            args.out,
            f"{feat_name}_mean",
            scaler.mean_.astype(np.float32),
        )
        write_hdf5(
            args.out,
            f"{feat_name}_scale",
            scaler.scale_.astype(np.float32),
        )


if __name__ == "__main__":
    main()
