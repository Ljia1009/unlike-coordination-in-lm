#!/usr/bin/env python3
"""
Downsampler script to randomly sample lines from train.corpus and filtered_dataset_all_unlike.txt
to match the line count of filtered_dataset_only_and.txt (2,144,648 lines).
"""

import argparse
import random
import sys
from pathlib import Path


def count_lines(file_path):
    """Count the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def read_all_lines(file_path):
    """Read all lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_sampled_lines(lines, output_path):
    """Write sampled lines to output file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def downsample_file(input_path, output_path, target_lines, seed=42):
    """
    Randomly downsample a file to the target number of lines.

    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_lines: Target number of lines to sample
        seed: Random seed for reproducibility
    """
    print(f"Processing {input_path}...")

    # Read all lines
    all_lines = read_all_lines(input_path)
    total_lines = len(all_lines)

    print(f"  Total lines: {total_lines:,}")
    print(f"  Target lines: {target_lines:,}")

    if total_lines <= target_lines:
        print(
            f"  Warning: Input file has {total_lines} lines, which is <= target {target_lines}")
        print(f"  Copying all lines to output...")
        write_sampled_lines(all_lines, output_path)
        return

    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly sample lines
    sampled_lines = random.sample(all_lines, target_lines)

    # Write sampled lines
    write_sampled_lines(sampled_lines, output_path)

    print(f"  Successfully sampled {target_lines:,} lines to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Downsample corpus files to match filtered_dataset_only_and.txt line count')
    parser.add_argument('--train-corpus', default='train.corpus',
                        help='Path to train.corpus file (default: train.corpus)')
    parser.add_argument('--filtered-all', default='../filtered_dataset_all_unlike.txt',
                        help='Path to filtered_dataset_all_unlike.txt file (default: ../filtered_dataset_all_unlike.txt)')
    parser.add_argument('--reference', default='../filtered_dataset_only_and.txt',
                        help='Path to reference file for line count (default: ../filtered_dataset_only_and.txt)')
    parser.add_argument('--output-train', default='train_downsampled.corpus',
                        help='Output path for downsampled train corpus (default: train_downsampled.corpus)')
    parser.add_argument('--output-filtered', default='../filtered_dataset_all_unlike_downsampled.txt',
                        help='Output path for downsampled filtered dataset (default: ../filtered_dataset_all_unlike_downsampled.txt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Check if reference file exists and get target line count
    if not Path(args.reference).exists():
        print(f"Error: Reference file {args.reference} does not exist")
        sys.exit(1)

    target_lines = count_lines(args.reference)
    print(f"Target line count from {args.reference}: {target_lines:,}")

    # Check if input files exist
    if not Path(args.train_corpus).exists():
        print(f"Error: Train corpus file {args.train_corpus} does not exist")
        sys.exit(1)

    if not Path(args.filtered_all).exists():
        print(
            f"Error: Filtered dataset file {args.filtered_all} does not exist")
        sys.exit(1)

    # Downsample train corpus
    downsample_file(args.train_corpus, args.output_train,
                    target_lines, args.seed)

    # Downsample filtered dataset
    downsample_file(args.filtered_all, args.output_filtered,
                    target_lines, args.seed)

    print("\nDownsampling completed successfully!")
    print(f"Output files:")
    print(f"  - {args.output_train}")
    print(f"  - {args.output_filtered}")


if __name__ == '__main__':
    main()
