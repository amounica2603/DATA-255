#!/bin/bash

# Download and setup shape predictor modele
echo "Downloading shape predictor model for mouth extraction and saving it to utility_models"
curl -L -o utility_models/shape_predictor_68_face_landmarks.dat https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat utility_models/
