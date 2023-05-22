#!/bin/bash

# Download and setup dataset
echo "Downloading the processed image CNN features to dataset/"
curl -c cookies.txt -s -L "https://drive.google.com/uc?export=download&id=1TBdIMgtPOzI11jGI6MOjQQoc90s4sUWV" > /dev/null
curl -L -o dataset/MIRACL_Processed_cnn_features.zip -b cookies.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' cookies.txt)&id=1TBdIMgtPOzI11jGI6MOjQQoc90s4sUWV"
rm cookies.txt

# Unzipping the dataset
echo "Unzipping the data to dataset to dataset/MIRACL_Processed_cnn_features/"
unzip dataset/MIRACL_Processed_cnn_features.zip -d dataset/ &> /dev/null
