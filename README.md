**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
ChessVision**

* Manvi Agarwal [LinkedIn](https://www.linkedin.com/in/manviagarwal27/)
* Adithya Rajeev [LinkedIn](https://www.linkedin.com/in/adithyar262/)
* Kevin Dong [LinkedIn](https://www.linkedin.com/in/xingyu-dong)

   Tested on: Jetson Orin, NVIDIA Ampere 8GB and RealSense Camera



# ChessVision

## Overview 

Chess Vision is a real-time chess board detection and chess piece classification model using edge device(Jetson Orin). Once the chess pieces per square are classified, we can generate FEN string which can be used to digitize the board on screen. 

![](images/demo_gif.gif)

The existing solution for real-time over the board games use electronic boards which are expensive and can be difficult to carry around. With Chess popularity increasing, having an inexpensive tool for live digitization of chess board will be useful. 

As of now, the achieved accuracy for board detection is:
- 95% accuracy in chessboard detection1
- 96% accuracy in piece classification1
- 20 second processing time per position1


## Key Features

- Automatic chessboard detection from images
- Chess piece classification using optimized CNNs
- FEN notation output
- Optimized for Nvidia Jetson Orin platform

Key features include real-time analysis, smooth visuals, and optimal resource utilization. The project will be implemented using CUDA, OpenCV, cuDNN, and TensorRT.

## Milestone Presentations
- [Milestone 1](https://docs.google.com/presentation/d/1U8ps8ubOPQaQodlSa4sc5Am6RymVd1RFZkvMr81juZM/edit?usp=sharing)
- [Milestone 2](https://docs.google.com/presentation/d/1VHaGN9LacqWEvyHqry0u9dCRp4Tq3KSkKzSpWPdE-mc/edit?usp=sharing)
- [Milestone 3](https://docs.google.com/presentation/d/1p_hStFTQr3upH9vLvdIlaWQbT13kGrGzDTNNHWYOgtw/edit?usp=sharing)

## Reference
- LiveChess2FEN (Original Project): [Link](https://developer.nvidia.com/blog/jetson-project-of-the-month-livechess2fen-provides-real-time-game-analysis/)
