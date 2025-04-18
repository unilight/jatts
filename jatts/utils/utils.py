# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import csv
import fnmatch
import logging
import os
import re
import sys
import tarfile
from distutils.version import LooseVersion

import h5py
import numpy as np
import torch
import yaml
from filelock import FileLock
import librosa


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_csv(path, dict_reader=False, lazy=False):
    """

    If `dict_reader` is set to True, then return <list, fieldnames>.
    If `dict_reader` is set to False, then return <list>.
    """

    """Read the csv file.

        Args:
            path (str): path to the csv file
            dict_reader (bool): whether to use dict reader. This should be set to true when the csv file has header.
            lazy (bool): whether to read the file in this funcion.
        
        Return:
            contents: reader or line of contents
            fieldnames (list): header. If dict_reader is False, then return None.

    """

    with open(path, newline="") as csvfile:
        if dict_reader:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
        else:
            reader = csv.reader(csvfile)
            fieldnames = None

        if lazy:
            contents = reader
        else:
            contents = [line for line in reader]

    return contents, fieldnames


def write_csv(data, path):
    """Write data to the output path.

    Args:
        path (str): path to the output csv file
        data (list): a list of dicts

    """
    fieldnames = list(data[0].keys())
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in data:
            writer.writerow(line)


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def load_model(checkpoint, config, stats=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """

    # lazy load for circular error
    import seq2seq_vc.models

    # get model and load parameters
    model_class = getattr(seq2seq_vc.models, config["model_type"])
    model = model_class(**config["model_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    return model

def read_audio(wav_path, sr, start=None, end=None, global_gain_scale=1.0):
    """Read audio file."""

    if start is not None and end is not None:
        offset = float(start)
        duration = float(end) - float(start)
    else:
        offset = 0.0
        duration = None

    # read, and resample if needed
    audio, fs = librosa.load(
        wav_path,
        sr=sr,
        offset=offset,
        duration=duration,
    )

    # check
    assert len(audio.shape) == 1, f"{wav_path} seems to be multi-channel signal."
    assert (
        np.abs(audio).max() <= 1.0
    ), f"{wav_path} seems to be different from 16 bit PCM."

    # apply global gain
    audio *= global_gain_scale
    if np.abs(audio).max() > 1.0:
        logging.warn(
            f"{wav_path} causes clipping (max: {np.abs(audio).max()}). "
            "it is better to re-consider global gain scale."
        )
        return None

    return audio