#!/bin/bash

baseline=$1
reverse=${2:-0}

# Base directory containing scene directories
scene_dir="offline-dataset/mp3d-dataset/900/val"

# Ensure the directory exists
if [ ! -d "$scene_dir" ]; then
    echo "Directory $scene_dir not found."
    exit 1
fi

# Read all scene names from the directory
scenes=($(ls $scene_dir))  # Assuming directory names directly correspond to scene names

# Reverse the scenes if reverse is set to 1
if [ "$reverse" -eq 1 ]; then
    # Reverse array using Bash array operations
    for (( idx=${#scenes[@]}-1 ; idx>=0 ; idx-- )) ; do
        reversed_scenes+=("${scenes[idx]}")
    done
    scenes=("${reversed_scenes[@]}")
fi

unset all_proxy; unset ALL_PROXY

# Iterate over each scene and execute the python script
for scene in "${scenes[@]}"; do
    # Create an output suffix by removing 'FloorPlan_Train' from the scene name
    # output_suffix=$(echo $scene | sed 's/FloorPlan_Train//')
    echo "Processing scene: $scene"
    python scripts/inference_habitat_transformer.py --scene_name $scene --baseline $baseline --mode ""
done

