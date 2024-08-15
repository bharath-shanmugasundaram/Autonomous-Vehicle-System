# Autonomous Vehicle System

This repository contains a series of projects developed as part of a self-driving car challenge. Each project focuses on a specific aspect of autonomous vehicle technology, including lane detection, vehicle detection, traffic sign classification, behavioral cloning, and various control and sensor fusion techniques.

## Project List & Goals

1. **Lane Finding Basic**
   - **Goal:** Create a simple pipeline to detect road lines in a frame taken from a roof-mounted camera.

2. **Traffic Sign Classifier**
   - **Goal:** Build a Convolutional Neural Network (CNN) in TensorFlow to classify traffic sign images from the Traffic Sign Dataset.

3. **Behavioral Cloning**
   - **Goal:** Train a deep neural network to replicate human steering behavior while driving. The network takes as input frames from the frontal camera and predicts the steering direction.

4. **Advanced Lane Finding**
   - **Goals:**
     - Compute the camera calibration matrix and distortion coefficients using chessboard images.
     - Apply distortion correction to raw images.
     - Use color transforms and gradients to create a thresholded binary image.
     - Apply a perspective transform to create a bird's-eye view of the lane.
     - Detect lane pixels and fit to find the lane boundary.
     - Determine lane curvature and vehicle position relative to the center.
     - Warp the detected lane boundaries back onto the original image.
     - Output a visual display of the lane boundaries along with numerical estimates of lane curvature and vehicle position.

5. **Vehicle Detection**
   - **Goal:** Develop a pipeline to reliably detect cars in a video captured from a roof-mounted camera.

6. **Extended Kalman Filter**
   - **Goal:** Implement the extended Kalman filter in C++ to track the position and velocity of a bicycle using simulated lidar and radar measurements.

7. **Unscented Kalman Filter**
   - **Goals:**
     - Implement the full processing chain (prediction, laser update, radar update).
     - Ensure the project runs in three modes: laser only, radar only, and both sensors.
     - Ensure RMSE (Root Mean Square Error) and NIS (Normalized Innovation Squared) meet the required thresholds.

8. **Kidnapped Vehicle**
   - **Goal:** Implement a 2D particle filter in C++ to localize a kidnapped vehicle using a map, initial GPS data, and noisy sensor/control data.

9. **PID Control**
   - **Goal:** Implement a Proportional-Integral-Derivative (PID) controller to keep the car on track by adjusting the steering angle.

10. **MPC Control**
    - **Goal:** Implement a Model Predictive Control (MPC) algorithm to optimize the cost function for steering control.

11. **Path Planning**
    - **Goal:** Safely navigate a virtual highway with other traffic. The car should maintain speed close to 50 MPH, avoid collisions, stay in the lane, and complete a loop around the highway.

12. **Road Segmentation**
    - **Goal:** Use a Fully Convolutional Network (FCN) to label road pixels in images.

13. **Traffic Light Classifier**
    - **Goal:** Integrate a traffic light classifier into the autonomous vehicle system.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- C++ Compiler
- Other required libraries listed in each project's `requirements.txt` file.

### Installation

Clone the repository and navigate to the specific project folder to get started:

```bash
git clone https://github.com/bharath-shanmugasundaram/autonomous-vehicle-system.git
cd autonomous-vehicle-system
