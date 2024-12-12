# Indian Sign Language Recognition - Building a vector embedding for each feature vector

## Introduction

This project implements an Indian Sign Language (ISL) recognition pipeline, designed to automatically interpret ISL gestures from video inputs and translate them into sign language text or symbols. The project is based on deep learning and computer vision techniques, employing a multi-step process to accurately recognize and output sign language gestures.

![ISL Pipeline](<static/ISL Overview.png>)


## Data Preparation 

### 1.  Video to Key Point Conversion
We converted video of sign gestures into Image Frames for model training and gesture recognition. Below is an outline of the data preparation steps, as demonstrated in the diagram:

![alt text](<static/ISL video - key.png>)


1. **Video Input**: The dataset consists of videos of individuals performing ISL gestures. These videos serve as the raw input data for the pipeline.
   - **Video Format**: The video data is in MP4 format.
   - **Dataset Source**: Videos are captured specifically for this project.

2. **Frame Extraction**:
   - Each video is broken down into individual frames. A frame is a still image taken at a specific time from the video sequence. 
   - The original frame shows a person making a hand gesture, with one hand raised near the face, indicating a sign, which is then passed into the next steps of the pipeline.

3. **Key Point Detection**:
   - Key points representing joints and important regions such as hand landmarks are detected in the image frames. 
   - As shown in the image, a layer of key points (marked in blue) is added on top of the image, capturing hand and body positions essential for identifying the gesture.

4. **Key Points Visualization**:
   - The final output shows only the detected key points on a plain white background, providing a simplified visual representation of the sign gesture. These points will be used by the model for sign recognition and further processing.

### 2. Key Points to n-Dimensional Label Conversion

The image illustrates the Vector Model used for sign language recognition, where image frames are converted into detailed labels representing hand and arm features.

![alt text](<static/ISL vector model.png>)

1. Image Frames: Two frames capture distinct hand gestures.
2. Vector Model: Processes these frames to extract specific attributes like:
   - Arm/Forearm/Elbow Orientation (folded, horizontal, diagonal)
   - Hand Position (abdomen, face)
   - Fingers (joined/not, fingertip orientation)
   - Palm Position (towards body)
   - Finger Closeness to Face (none, hair)


## Dataset Directory Structure
The Dataset contains the model v1 inference set, Train and the Test set

```
/Dataset/
├── modelv1_inference
├── Test
│   ├── Multiple_person_4715.csv
│   ├── test_dataset.csv
│   └── test_dataset_v2.csv
└── Train
    ├── v1_10k_normalized 
    └── v2_4.7k_normalized
```
* `modelv1_inference` : Contains CSV files with model predictions for each image. These predictions represent the output of the inference stage, where the model attempts to recognize ISL gestures based on the input data.

* `Test` : Contains test datasets used to validate the model.
  
  - `Multiple_person_4715.csv`: A test dataset derived from 5407 samples of `test_dataset.csv` after removing blur frames.
  - `test_dataset.csv` : The original test dataset used for model validation which contains 5407 samples of image frames from 200 videos.
  - `test_dataset.csv` : The test dataset used for model validation which contains 2001 samples of image frames from 200 videos.

* `Train` : Contains normalized versions of the training datasets used to train the model.
  
  - `v1_10k_normalized` : A version of the dataset that includes 10,000 samples added to the original v0 version.  
  - `v2_4.7k_normalized` : A version of the dataset that includes 4,700 samples added to the v1 version.

### 10k_manual_annotation

Contains CSV files with image and its associated ground truth(?) label(?). These annotations are used for training and evaluating the model’s performance in recognizing ISL gestures.

### vector_models
`jesintha_v1` : Model built with Jesintahs's video (images from 25 signs videos). The data was normalised (upsampling and downsampling) with respect to the classes. Blurred frames are removed. 

`10kmodels_v2` : Models built with 10,000 non blur images taken from dataset_v1 (2000 videos containing 10 signs) 

`multi_people_models_v3` : Models built with 4.7k non blur images from `dataset_v1` (2k images comprising of 10 signs).
Image selection - almost 3 frames and 1 frame per video was selected. 

### sign_models

## Contact Information
For any questions or further details, please reach out to the project maintainer at

    1. Stephen L 
    2. Sandeep Kumar Suresh