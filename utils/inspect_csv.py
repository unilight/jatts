#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

"""
Analyze duration and speaker statistics from a CSV file.
"""

import pandas as pd
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze duration and speaker statistics from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--speaker_col", type=str, default="spk", help="Column name to use for speaker identification (default: 'spk')")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)

    # Calculate duration
    df['duration'] = df['end'] - df['start']


    # (1) Overall duration statistics
    total_duration = df['duration'].sum()
    print("=== Duration Statistics (Overall) ===")
    print(f"Count: {len(df):.0f}")
    print(f"Total Duration: {total_duration:.2f} seconds ({total_duration / 3600:.2f} hours)")
    print(f"Mean: {df['duration'].mean():.2f}")
    print(f"Min: {df['duration'].min():.2f}")
    print(f"Max: {df['duration'].max():.2f}")
    print(f"Std: {df['duration'].std():.2f}")

    # (2) Speaker statistics (if speaker column exists)
    if args.speaker_col in df.columns:
        num_speakers = df[args.speaker_col].nunique()
        print(f"\n=== Number of Unique Speakers ===\n{num_speakers}")

        speaker_groups = df.groupby(args.speaker_col)

        # Samples per speaker
        samples_per_speaker = speaker_groups.size()
        print("\n=== Samples Per Speaker Statistics ===")
        print(f"Mean: {samples_per_speaker.mean():.2f}")
        print(f"Min: {samples_per_speaker.min():.2f}")
        print(f"Max: {samples_per_speaker.max():.2f}")
        print(f"Std: {samples_per_speaker.std():.2f}")

        # Duration per speaker
        duration_per_speaker = speaker_groups['duration'].sum()
        print("\n=== Total Duration Per Speaker Statistics ===")
        print(f"Mean: {duration_per_speaker.mean():.2f}")
        print(f"Min: {duration_per_speaker.min():.2f}")
        print(f"Max: {duration_per_speaker.max():.2f}")
        print(f"Std: {duration_per_speaker.std():.2f}")
        print(f"Total Duration Across All Speakers: {duration_per_speaker.sum():.2f} seconds ({duration_per_speaker.sum() / 3600:.2f} hours)")
    else:
        print(f"\n[Warning] Speaker column '{args.speaker_col}' not found in the CSV. Skipping speaker statistics.")

if __name__ == "__main__":
    main()