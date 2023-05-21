#!/bin/bash

# Download and setup dataset
echo "Downloading the processed image CNN features to data/"
curl -c cookies.txt -s -L "https://drive.google.com/uc?export=download&id=1Nt0ZGCiPzGk3sVuoNjF26kz7VnxDulUq" > /dev/null
curl -L -o data/MIRACL-VC1_all_in_one.zip -b cookies.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/download/ {print $NF}' cookies.txt)&id=1TcPVwcGcFFi98_oNZJBJNUnRii4qCRwx"
rm cookies.txt

# Unzipping the dataset
echo "Unzipping the data to dataset to data/"
unzip data/MIRACL-VC1_all_in_one.zip -d data/MIRACL_Processed &> /dev/null

https://drive.google.com/file/d//view?usp=share_link