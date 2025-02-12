#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
from pathlib import Path

import kaldiio
import librosa
import matplotlib
import numpy as np
from jatts.utils import read_csv
from jatts.utils.signal import world_extract
from joblib import Parallel, delayed

matplotlib.use("Agg")  # noqa #isort:skip
import matplotlib.pyplot as plt  # noqa isort:skip


def create_histogram(
    data, figure_path, range_min=-70, range_max=20, step=10, xlabel="Power [dB]"
):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    plt.hist(
        data,
        bins=200,
        range=(range_min, range_max),
        density=True,
        histtype="stepfilled",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()


def extract_f0_and_npow(item, f0min=40, f0max=500):
    """
    F0 and npow extraction

    Parameters
    ----------
    item : dict,
        item row in csv file

    Returns
    -------
    dict :
        Dictionary consisting of F0 and npow arrays

    """
    x, fs = librosa.load(item["wav_path"], sr=None)
    return item, world_extract(x, fs, f0min, f0max)


def main():
    dcp = "Create histogram for speaker-dependent configure"
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument("--n_jobs", type=int, default=16, help="# of CPUs")
    parser.add_argument("--csv", type=str, default=None, help="path to csv file")
    parser.add_argument("--figure_dir", type=str, help="Directory for figure output")
    args = parser.parse_args()

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # read dataset
    dataset, _ = read_csv(args.csv, dict_reader=True)

    # get speaker list
    if "spk" not in dataset[0]:  # single speaker
        num_spks = 1

    # extract features in parallel
    logging.info("Extracting features in parallel")
    results = Parallel(n_jobs=args.n_jobs)(
        [delayed(extract_f0_and_npow)(item) for item in dataset]
    )

    if num_spks == 1:
        f0histogrampath = os.path.join(args.figure_dir, "f0histogram.png")
        npowhistogrampath = os.path.join(args.figure_dir, "npowhistogram.png")

        f0s = [r["f0"] for _, r in results]
        npows = [r["npow"] for _, r in results]

        # stack feature vectors
        f0s = np.hstack(f0s).flatten()
        npows = np.hstack(npows).flatten()

        # create a histogram to visualize F0 range of the speaker
        create_histogram(
            f0s,
            f0histogrampath,
            range_min=40,
            range_max=700,
            step=50,
            xlabel="Fundamental frequency [Hz]",
        )

        # create a histogram to visualize npow range of the speaker
        create_histogram(
            npows,
            npowhistogrampath,
            range_min=-70,
            range_max=20,
            step=10,
            xlabel="Frame power [dB]",
        )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
