#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, help="Path to save accelerate config file")
    parser.add_argument("--num_processes", type=int, required=True)
    parser.add_argument("--num_machines", type=int, required=True)
    parser.add_argument("--machine_rank", type=int, required=True)
    parser.add_argument("--main_process_ip", required=True)
    parser.add_argument("--main_process_port", type=int, default=29500)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    args = parser.parse_args()

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "machine_rank": args.machine_rank,
        "main_training_function": "main",
        "main_process_ip": args.main_process_ip,
        "main_process_port": args.main_process_port,
        "num_machines": args.num_machines,
        "num_processes": args.num_processes,
        "mixed_precision": args.mixed_precision,
        "use_cpu": False
    }

    os.makedirs(os.path.dirname(args.config_path), exist_ok=True)
    with open(args.config_path, "w") as f:
        yaml.dump(config, f)

    print(f"[INFO] Accelerate config saved to {args.config_path}")

if __name__ == "__main__":
    main()