#!/bin/bash

# download raw dataset
./utility_scripts/download_raw_dataset.sh

# Download preprocessed data
./utility_scripts/download_processed_dataset.sh

# Download features data
./utility_scripts/download_processed_features.sh

# Download required utility models
./utility_scripts/download_utility_models.sh