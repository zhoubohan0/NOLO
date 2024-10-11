#!/bin/bash


# Assign arguments to variables
baseline=$1
ckpt_file=$2
mode=$3
context_type=$4
seed=${5:-42} 

# Base directory containing scene directories
scene_dir="offline-dataset/mp3d-dataset/900/val"

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
    # Create an output suffix by removing 'FloorPlan_Train' from the scene name
    # output_suffix=$(echo $scene | sed 's/FloorPlan_Train//')
    # --output_dir "${ckpt_dir}" \
    echo "Processing scene: $scene"
    python scripts/inference_habitat_transformer.py \
        --baseline $baseline \
        --scene_name $scene \
        --mode $mode\
        --context_type $context_type\
        --ckpt_file $ckpt_file\
        --seed $seed
done

