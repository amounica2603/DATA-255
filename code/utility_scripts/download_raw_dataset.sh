#!/bin/bash

# Download and setup dataset
echo "Downloading the Dataset to dataset/"
curl -c cookies.txt -s -L "https://drive.google.com/uc?export=download&id=1TcPVwcGcFFi98_oNZJBJNUnRii4qCRwx" > /dev/null
curl -L -o dataset/MIRACL-VC1_all_in_one.zip -b cookies.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' cookies.txt)&id=1TcPVwcGcFFi98_oNZJBJNUnRii4qCRwx"
rm cookies.txt

# Unzipping the dataset
echo "Unzipping the data to dataset to dataset/MIRACL-VC1_all_in_one/"
unzip dataset/MIRACL-VC1_all_in_one.zip -d dataset/MIRACL-VC1_all_in_one &> /dev/null
rm dataset/MIRACL-VC1_all_in_one/calib.txt
