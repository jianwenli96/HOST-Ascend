#!/bin/bash

# Step 1: build the task dictionary (HDF5) from raw episode directories
python build_task_dictionary.py \
  --input_dir /path/to/your/datasets \
  --output_path ./output/dataset_name.hdf5 \
  --dataset_name dataset_name \
  --clear

# Step 2: write task_paths.json into each episode folder
python write_task_paths.py \
  --hdf5_path ./output/dataset_name.hdf5