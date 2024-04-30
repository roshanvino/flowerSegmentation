# Flower Segmentation Image Processing Pipeline

This repository contains the Python code for an Image Processing Project assigned in the Introduction to Image Processing module at the School of Computer Science, University of Nottingham.

## Overview

This project aims to develop an image processing pipeline to separate flower material from the background in images. The pipeline includes steps such as color space conversion, noise reduction, thresholding/segmentation, and binary image processing.

## Dataset

The dataset consists of images of three species of flowers categorized into three sub-folders based on the complexity of foreground/background features (easy, medium, and hard). Ground truth for each image is provided for evaluation purposes.

## Requirements

- Python 
- OpenCV
- NumPy
- Matplotlib

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main Python program `flowerSeg.py`.
4. The segmented images will be saved in the `output` folder categorized into easy, medium, and hard sub-folders.

## File Structure

- `flowerSeg.py`: Main Python program for flower segmentation.
- `dataset/`: Folder containing all Image files
  - `ground_truths/`: Folder containing the result images we are aiming for
  - `images/`: Folder containing input images.
    - `easy/`
    - `medium/`
    - `hard/`
  - `output/`: Folder containing segmented images generated after processing.

## Results

For details on the method employed, results obtained, and critical evaluation, please refer to the conference paper `COMP2005-CW-GroupXXX.pdf`.
