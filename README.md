# Face Recognition Web App

This web application allows users to utilize their webcam to perform real-time face recognition. The app is built using Flask, OpenCV, and FaceNet.

## Features

- Real-time face detection and recognition using the webcam.
- Face embeddings extraction and recognition with SVM model.
- Styling using Tailwind CSS for a modern and responsive look.

## Prerequisites


Before running the application, make sure you have the following installed:

- Python 3.x
- OpenCV (cv2)
- Flask
- NumPy
- scikit-learn
- keras-facenet

Install the required Python packages using pip:

```bash
pip install opencv-python flask numpy scikit-learn keras-facenet

Usage
Click the "Start Webcam" button to open the webcam and start face recognition.
The app will detect faces in real-time and attempt to recognize them using pre-trained face embeddings and an SVM model.
Detected faces will be highlighted with a green rectangle, and the recognized name (if available) will be displayed above the rectangle.
If a face cannot be recognized with high confidence, it will be labeled as "Unknown."

Copyright © [Syed Farhat Hussain Shah , Hassan Ali] [2023]. All rights reserved.
...
