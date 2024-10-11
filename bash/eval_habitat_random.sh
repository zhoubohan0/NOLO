#!/bin/bash

# Base directory containing scene directories
scene_dir="./offline-dataset/mp3d-dataset/900/val"

# Ensure the directory exists
# if [ ! -d "$scene_dir" ]; then
#     echo "Directory $scene_dir not found."
#     exit 1
# fi

# Read all scene names from the directory
scenes=($(ls $scene_dir))  # Assuming directory names directly correspond to scene names

# Iterate over each scene and execute the python script
for scene in "${scenes[@]}"; do
    # Create an output suffix by removing 'FloorPlan_Train' from the scene name
    # output_suffix=$(echo $scene | sed 's/FloorPlan_Train//')
    echo "Processing scene: $scene"
    python scripts/inference_habitat_transformer.py \
        --scene_name $scene \
        --output_dir "logs/lograndom-mp3d" \
        -r 1\
        --seed 100
done

