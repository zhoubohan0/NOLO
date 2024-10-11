#!/bin/bash

baseline=$1
ckpt_file=$2
mode=$3
context_type=$4
seed=${5:-42}  # Set default seed value to 42 if not provided

# Base directory containing scene directories
scene_dir="offline-dataset/robothor-dataset/900/val"

# Ensure the directory exists
if [ ! -d "$scene_dir" ]; then
    echo "Directory $scene_dir not found."
    exit 1
fi

# Read all scene names from the directory
scenes=($(ls $scene_dir))  # Assuming directory names directly correspond to scene names

# Get the directory name of the checkpoint file
ckpt_dir=$(dirname "$ckpt_file")

# Iterate over each scene and execute the python script
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"
    python scripts/inference_robothor_transformer.py \
        --scene_name $scene \
        --ckpt_file $ckpt_file \
        --mode $mode \
        --context_type $context_type \
        --baseline $baseline \
        --seed $seed
done