# Smart Checkout System

This project implements a smart checkout system using YOLO V8 object detection to track and calculate the prices of products crossing a designated checkout line in a video feed.

## Overview

The system uses the YOLO V8 model for real-time object detection and tracking. It identifies products as they move through a video frame. The system then calculates the total cost of detected products based on predefined prices.

## Features

- Object detection and tracking using YOLO V8.
- Cost calculation for specific product classes as they cross a defined line on the screen.
- Visual annotation of detected objects on the video frames.
- Output video with annotated frames showing tracked objects and total cost.

## Components

### 1. Python Libraries Used

- `ultralytics`: YOLO V8 model for object detection and tracking.
- `cv2` (OpenCV): For video capture, frame manipulation, and annotation.
- `supervision`: Custom library/module for video processing and output.

### 2. Files

- `main.py`: Python script implementing the smart checkout system.
- `utils.py`: Utility functions including `get_product_cost` for calculating product costs.
- `model/yolov8n.pt`: Pre-trained YOLO V8 model file.
- `test_videos/`: Directory containing input video files for testing.
- `test_output/`: Directory for storing output videos with annotated frames.

### 3. Usage

To run the smart checkout system:

1. Ensure all dependencies are installed (`ultralytics`, `cv2`, `supervision`).
2. Place the input video (`video4.mp4` or other test videos) in the `test_videos/` directory.
3. Run `main.py` to process the video and generate the output.

![alt text](output4.gif)
