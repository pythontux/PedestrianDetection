# PedestrianDetection
This repository contains the code used for a project on Computer Vision in EPI Gijón.
Code by Álvaro Gregorio Gómez

OpenCV 2.49 libraries are required to compile the project.
Once compiled, the program process a sequence of images, computing the u and v-disparity images. After that, we extract lines that represent the obstacles in these two images by Probabilistic Hough Transform. Finally, obstacles are marked by bounding boxes
