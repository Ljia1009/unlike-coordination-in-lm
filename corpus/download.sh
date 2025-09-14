#!/bin/bash

set -euo pipefail # automatically exit if any program errors, prohibit undefined variables

BASE_URL="https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/"
CHOICES="{train/valid/test/vocab}"
EXT=".corpus"

if [[ $# -ne 1 && $# -ne 2 ]]; then
    echo "Usage: download.sh $CHOICES [/path/to/out/directory]"
else
    if [[ $# -eq 2 ]]; then
        OUT_DIR="$2"
    else
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
        OUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
    fi

    mkdir -p "$OUT_DIR"
    OUT_DIR="$(cd "$OUT_DIR" && pwd -P)"

    if [[ $1 = "train" || $1 = "valid" || $1 = "test" || $1 = "vocab" ]]; then
        CORPUS="$1"
        OUT_FILE="$OUT_DIR/$CORPUS$EXT"
        URL="${BASE_URL}${1}.txt"
        echo "Downloading $1 corpus..."
        echo "Downloading from: $URL"
        echo "Downloading to: $OUT_FILE"
        curl -fsSL "$URL" -o "$OUT_FILE"
    else
        echo "Invalid corpus download type: $1. Please select one of: $CHOICES"
    fi
fi