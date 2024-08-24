# APTOS_Blindness_Detection
https://www.kaggle.com/competitions/aptos2019-blindness-detection

# Diabetic Retinopathy Classification using Neural Networks (CNN)

This repository contains code for classifying diabetic retinopathy using convolutional neural networks (CNNs). The model is trained on the APTOS 2019 dataset and predicts the severity of diabetic retinopathy from retinal images.

## Overview

Diabetic retinopathy is a medical condition where the walls of the blood vessels in the retina weaken. It is crucial to classify the severity of this condition to provide timely treatment and prevent vision loss. The classification task involves categorizing images into one of the five stages of diabetic retinopathy:

- 0 - No DR
- 1 - Mild
- 2 - Moderate
- 3 - Severe
- 4 - Proliferative DR

# Dataset

The dataset issued for the aptos competition 2019 consisted of 
- 3662 labelled train images with labels in the csv file for training and 
- 1928 unlabelled images for prediction. 

Initially the data is read mapped with labels in the csv file and then the read csv files are split into train, test and validation splits. The ratio of the split was maintained at test size = 0.3. 
Stratify based on labels is used to maintain the mix or proportion of all categories are present in all the splits.

# Open and Load Images or dataset

Based on the observation, the csvs contain the labels and the photo ids. 
- Train test split is performed in the csvs and then their respective values can be mapped with the id in the csv file.
- Here I have used the open cv library to read the images and then albumentations library to transform the images.

**DatasetClass**
- Firstly the id and label are tapped from the csv file, then the id is sent along with the file format and is concatenated as string along with the data directory followed by file format like jpeg, png etc. 
- Then the cv2 command 'imread' is used to open the images and convert them to arrays. 

**Preprocess**
- Then the format is changed from bgr to rgb which is its default return type. Also in some grayscale images are used. 
- Then the respective labels are extracted and are attac
- Then post opening the image it is sent to the transforms method and then brought back in. 
- The return value of the method contains image tensors and which can be later unpacked if needed.

**DataLoader**
- The processed data from the above step are passed through data loader to split into batches and shuffle the dataset to add some more randomness.
- Post the data loader the output was in the form of list of tensors of all batches input data, tensors of all labels.
Here the parameters used are batch size 1 for most models. Often batch sizes are kept low to 

# Experiments Table

- Initially we did all right and were doing it with a shuffle in the prediction dataset, which lead to mismatch of labels and ids. And the kappa scores were also too low like 0.0089 in submissions both public and private. Then it was found and changed to get better results as below.

So firstly the data is trained with minimum preprocessing like converting to tensors and then experimented further with augmentations and different other possibilities.

# With Basic Augmentations

| Trial | Model                | Experiment | Size | Epochs | Train Kappa | Test Kappa | Validation Kappa | Private Score | Public Score |
| ----- | -----                | ---------- | ---- | ------ | -------------- | ------------- | ------------------- | ------------ | ------------- |
|   Best   | SeResNext50_32x4d | Resizing, tensor, sharpen + least loss | 512 | 10 | 0.74217 | 0.69401 | 0.67644 | 0.863467 | 0.649313 |
|   1   | SeResNext26d_32x4d   | Resizing, tensor + best kappa | 224 | 9 | 0.847368 | 0.83590 | 0.82536 | 0.854206 | 0.660446 |
|   2   | tf efficientnet v2b3 | Resizing, tensor, sharpen | 512 | 6 | 0.76123 | 0.77956 | 0.75432 | 0.836953 | 0.681245 |
|   3   | SeResNext26d_32x4d   | Resizing, tensor, sharpen + least loss | 224 | 11 | 0.77723 | 0.78915 | 0.737066 | 0.835079 | 0.660425 |
|   4   | Inception_resnet_v2  | Resizing, tensor, sharpen + least loss | 512 | 11 | 0.78424 | 0.77248 | 0.742394 | 0.826997 | 0.667065 |
|   5   | Efficientnet b0      | Resizing and convert to tensor | 512 | 15 | 0.83469 | 0.88636 | 0.880118 | 0.82232 | 0.560797 |
|   6   | Inception V3         | Resizing, tensor, sharpen + best kappa | 512 | 7 | 0.83415 | 0.88769 | 0.88004 | 0.816187 | 0.528484 |
|   7   | ResNet50             | Resizing and convert to tensor | 512 | 15 | 0.75954 | 0.87090 | 0.872545 | 0.78086 | 0.468281 |
|   8   | SEResNet101          | Resizing and convert to tensor | 224 | 10 | 0.764354 | 0.694564 | 0.704622 | 0.436233 | 0.015804 |
|   9   | EfficientNet_b0      | Resizing and convert to tensor, consider as regression | 224 | 10 | 0.764354 | 0.694564 | 0.704622 | 0.436233 | 0.015804 |

# With Augmentations

| Trial | Model                | Experiment | Size | Epochs | Train Kappa | Test Kappa | Validation Kappa | Private Score | Public Score |
| ----- | -----                | ---------- | ---- | ------ | -------------- | ------------- | ------------------- | ------------ | ------------- |
|   1   | SeResNext26d_32x4d | Resizing, tensor, sharpen + least loss | 224 | 15 | 0.72193 | 0.76100 | 0.781142 | 0.727504 | 0.57019 |
|   2   | Inception V3         | Resize, gaussian blur + least loss  | 512 | 12 | 0.78737 | 0.73883 | 0.72617 | 0.779498 | 0.528283 |
|   3   | tf efficientnet v2b3 | Resizing, tensor, sharpen | 224 | 7 | 0.71839 | 0.699702 | 0.705932 | 0.750287 | 0.573132 |
|   4   | ResNet50             | Resizing, tensor, Gaussian blur, sharpen | 224 | 12 | 0.71885 | 0.75661 | 0.75658 | 0.577191 | 0.112045 |
|   5   | SeResNet50_32x4d | Resizing, tensor, sharpen, brightness, contrast, hue, blurs | 224 | 15 | 0.67938 | 0.75522 | 0.74260 | 0.380669 | 0.264965 |

# Inclusion of 2015 data

Since many discussion suggested and also upon knowledge adding up of data probably from a good source might improve performance of the model in some cases hence gave it a try by adding data of the 2015 competition.

| Trial | Model                | Experiment | Size | Epochs | Train Kappa | Test Kappa | Validation Kappa | Private Score | Public Score |
| ----- | -----                | ---------- | ---- | ------ | -------------- | ------------- | ------------------- | ------------ | ------------- |
|   1   | SeResNext50_32x4d | Resizing, tensor, sharpen + least loss | 224 | 15 | 0.507643 | 0.78652 | 0.77717 | 0.755948 | 0.445672 |
