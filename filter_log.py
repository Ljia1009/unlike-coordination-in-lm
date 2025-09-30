#!/usr/bin/env python3
"""
Script to filter out progress bar information from training logs,
keeping only evaluation info and custom log messages.
"""

import re
import sys


def is_progress_bar(line):
    """Check if a line is a progress bar."""
    # Progress bars typically contain percentage, bars, and timing info
    progress_patterns = [
        r'^\s*\d+%\|',  # Lines starting with percentage
        r'^\s*\d+/\d+\s+\[',  # Lines with step counts and timing
        r'^\s*\|.*\|.*\[.*\]',  # Lines with bars and timing
        r'^\s*[█▉▊▋▌▍▎▏\s]+\|',  # Lines with bar characters
    ]

    for pattern in progress_patterns:
        if re.search(pattern, line):
            return True
    return False


def is_important_content(line):
    """Check if a line contains important information to keep."""
    # Keep lines that contain:
    # - Custom log messages (not starting with numbers/percentages)
    # - Training metrics (containing loss, epoch, etc.)
    # - Evaluation metrics (containing eval_)
    # - Model saving messages
    # - Training completion messages

    if not line.strip():
        return False

    # Skip progress bars
    if is_progress_bar(line):
        return False

    # Keep important patterns
    important_patterns = [
        r'loss.*:',  # Training metrics
        r'eval_',    # Evaluation metrics
        r'epoch.*:',  # Epoch information
        r'grad_norm',  # Gradient norm
        r'learning_rate',  # Learning rate
        r'Tokenizer',  # Setup messages
        r'Loading',  # Setup messages
        r'Configuring',  # Setup messages
        r'Model created',  # Setup messages
        r'Training',  # Training messages
        r'Saving',   # Saving messages
        r'Checkpoint',  # Checkpoint messages
        r'Directory',  # Directory messages
        r'Contents',  # Directory contents
        r'No checkpoint',  # Checkpoint messages
        r'Starting',  # Starting messages
        r'Complete',  # Completion messages
        r'To use the model',  # Usage instructions
        r'loss_type',  # Configuration warnings
    ]

    for pattern in important_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    # Keep lines that look like JSON (training/eval metrics)
    if line.strip().startswith('{') and line.strip().endswith('}'):
        return True

    return False


def filter_log_file(input_file, output_file):
    """Filter the log file to remove progress bars."""
    kept_lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if is_important_content(line):
                kept_lines.append(line)

    # Write filtered content to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(kept_lines)

    print(f"Filtered log file created: {output_file}")
    print(f"Original lines: {line_num}")
    print(f"Kept lines: {len(kept_lines)}")
    print(f"Removed lines: {line_num - len(kept_lines)}")


if __name__ == "__main__":
    input_file = "/Users/jia/Documents/Thesis/experiments/logs/train_original_log"
    output_file = "/Users/jia/Documents/Thesis/experiments/logs/train_original_log_cleaned"

    filter_log_file(input_file, output_file)
