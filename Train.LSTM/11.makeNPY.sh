#!/bin/bash

# === Parameters ===
NUM_SPLITS=4  # Number of parallel processes (can be changed, depends upon RAM)
INPUT_FILE="allchains.txt"       # This file have to be the union of training and validatation chains sets
SCRIPT_NAME="LSTM_make_data.py"  # Your script name (replace with actual)
OUTPUT_DIR="output.LSTM"

# === Checks ===
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ File $INPUT_FILE not found!"
    exit 1
fi

# === Preparation ===
echo "Leftofer cleanup*"
rm -f part_*
rm -rf "${OUTPUT_DIR}_part_"*

# === Split into parts ===
echo "Splitting $INPUT_FILE into $NUM_SPLITS parts..."
split -n l/$NUM_SPLITS "$INPUT_FILE" part_

# === Start processes ===
i=0
PART_OUTPUT_DIRS=()
for PART in part_*; do
    PART_OUTPUT="${OUTPUT_DIR}_part_$i"
    mkdir -p "$PART_OUTPUT"
    PART_OUTPUT_DIRS+=("$PART_OUTPUT")
    echo "Processing $PART -> $PART_OUTPUT"
    python3 "$SCRIPT_NAME" "$PART" "$PART_OUTPUT" &
    ((i++))
done

wait
echo "All processes completed."

# === Concatenate results ===
echo "Combining results into $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"

for PART_DIR in "${PART_OUTPUT_DIRS[@]}"; do
    if [ -d "$PART_DIR" ]; then
        mv "$PART_DIR"/* "$OUTPUT_DIR"/
    fi
done

# === Cleanup temporary files ===
echo "Removing temporary parts and directories..."
rm -f part_*
rm -rf "${OUTPUT_DIR}_part_"*

echo "All is done! All results are collected in $OUTPUT_DIR."
