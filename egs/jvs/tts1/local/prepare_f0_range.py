#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_f0_path", type=str, required=True, help="original f0 range file path"
    )
    parser.add_argument("--out", type=str, required=True, help="out f0 yaml")

    args = parser.parse_args()

    # read original f0 range
    f0_all = {}
    with open(args.original_f0_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines[1:]:
            spk, gender, f0min, f0max = line.split(" ")
            f0_all[spk] = {
                "f0min": int(f0min),
                "f0max": int(f0max),
            }

    # write yaml
    with open(args.out, "w") as f:
        yaml.dump(f0_all, f, Dumper=yaml.Dumper)