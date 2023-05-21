#!/bin/bash

# Download and setup dataset
echo "Downloading the processed image CNN features to data/"
curl -c cookies.txt -s -L "https://drive.google.com/uc?export=download&id=1gwrAOvytGuR0j9UUH3b9kWJYs2P3kWAN" > /dev/null
curl -L -o data/MIRACL_Processed.zip -b cookies.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' cookies.txt)&id=1gwrAOvytGuR0j9UUH3b9kWJYs2P3kWAN"
rm cookies.txt

# Unzipping the dataset
echo "Unzipping the processed dataset to data/MIRACL_Processed/"
unzip data/MIRACL_Processed.zip -d data/MIRACL_Processed &> /dev/null
