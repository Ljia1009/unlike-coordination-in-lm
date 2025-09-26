#!/usr/bin/env python3
"""
Downsampler script to randomly sample lines from large inputs to a target size.

Supports:
- Using filtered_dataset_only_and.txt as target (default)
- Using filtered_dataset_all_unlike(.txt or _2.txt) as target
- Auto target = min(count(only_and), count(all_unlike))
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
        description='Downsample corpus files to a chosen target line count')
    parser.add_argument('--train-corpus', default='train.corpus',
                        help='Path to train.corpus file (default: train.corpus)')
    parser.add_argument('--filtered-all', default='../filtered_dataset_all_unlike.txt',
                        help='Path to filtered_dataset_all_unlike file (default: ../filtered_dataset_all_unlike.txt)')
    parser.add_argument('--only-and', default='../filtered_dataset_only_and.txt',
                        help='Path to filtered_dataset_only_and.txt (default: ../filtered_dataset_only_and.txt)')
    parser.add_argument('--reference', default=None,
                        help='Explicit reference file to force the target line count (overrides --mode)')
    parser.add_argument('--mode', choices=['only_and', 'all_unlike', 'auto_min', 'custom'], default='all_unlike',
                        help='Target selection: all_unlike (default), only_and, auto_min (min of the two), custom (use --reference)')
    parser.add_argument('--output-train', default='train_downsampled.corpus',
                        help='Output path for downsampled train corpus (default: train_downsampled.corpus)')
    parser.add_argument('--output-filtered', default='../filtered_dataset_all_unlike_downsampled.txt',
                        help='Output path when downsampling all_unlike (default: ../filtered_dataset_all_unlike_downsampled.txt)')
    parser.add_argument('--output-only-and', default='../filtered_dataset_only_and_downsampled.txt',
                        help='Output path when downsampling only_and (default: ../filtered_dataset_only_and_downsampled.txt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Resolve target based on mode/reference
    only_and_path = Path(args.only_and)
    all_unlike_candidates = [
        Path(args.filtered_all), Path(args.filtered_all_alt)]
    all_unlike_path = next(
        (p for p in all_unlike_candidates if p.exists()), None)

    if args.mode == 'custom':
        if not args.reference:
            print('Error: --mode custom requires --reference')
            sys.exit(1)
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"Error: Reference file {ref_path} does not exist")
            sys.exit(1)
        target_lines = count_lines(ref_path)
        print(f"Target line count from {ref_path}: {target_lines:,}")
    elif args.mode == 'only_and':
        if not only_and_path.exists():
            print(f"Error: Only-and file {only_and_path} does not exist")
            sys.exit(1)
        target_lines = count_lines(only_and_path)
        print(f"Target line count from {only_and_path}: {target_lines:,}")
    elif args.mode == 'all_unlike':
        if not all_unlike_path:
            print(
                f"Error: All-unlike file not found at {all_unlike_candidates[0]} or {all_unlike_candidates[1]}")
            sys.exit(1)
        target_lines = count_lines(all_unlike_path)
        print(f"Target line count from {all_unlike_path}: {target_lines:,}")
    else:  # auto_min
        missing = []
        if not only_and_path.exists():
            missing.append(str(only_and_path))
        if not all_unlike_path:
            missing.append(
                f"{all_unlike_candidates[0]} or {all_unlike_candidates[1]}")
        if missing:
            print("Error: Missing files for auto_min: " + ", ".join(missing))
            sys.exit(1)
        only_and_count = count_lines(only_and_path)
        all_unlike_count = count_lines(all_unlike_path)
        target_lines = min(only_and_count, all_unlike_count)
        print(
            f"Auto target (min): only_and={only_and_count:,}, all_unlike={all_unlike_count:,} -> target={target_lines:,}")

    # Check if input files exist
    if not Path(args.train_corpus).exists():
        print(f"Error: Train corpus file {args.train_corpus} does not exist")
        sys.exit(1)

    if not Path(args.filtered_all).exists() and not Path(args.filtered_all_alt).exists():
        print(
            f"Error: All-unlike dataset not found at {args.filtered_all} or {args.filtered_all_alt}")
        sys.exit(1)
    if not Path(args.only_and).exists():
        print(f"Error: Only-and dataset not found at {args.only_and}")
        sys.exit(1)

    # Downsample train corpus
    downsample_file(args.train_corpus, args.output_train,
                    target_lines, args.seed)

    # Downsample the opposite filtered dataset relative to the target
    if args.mode == 'all_unlike' or (args.mode == 'custom' and ref_path == Path(args.filtered_all)) or (args.mode == 'custom' and ref_path == Path(args.filtered_all_alt)):
        # Target is all_unlike → downsample only_and
        filtered_input = str(args.only_and)
        filtered_output = args.output_only_and
    else:
        # Target is only_and or auto_min/custom otherwise → downsample all_unlike
        filtered_input = args.filtered_all if Path(
            args.filtered_all).exists() else args.filtered_all_alt
        filtered_output = args.output_filtered

    downsample_file(filtered_input, filtered_output, target_lines, args.seed)

    print("\nDownsampling completed successfully!")
    print(f"Output files:")
    print(f"  - {args.output_train}")
    print(f"  - {args.output_filtered}")


if __name__ == '__main__':
    main()
