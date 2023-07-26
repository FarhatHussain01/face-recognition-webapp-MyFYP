import cv2 as cv
import numpy as np
import os
from flask import Flask, render_template, Response
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet and load face embeddings
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_6classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load Haar cascade classifier for face detection
haarcascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load SVM model for face recognition
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

app = Flask(__name__)

def gen():
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = rgb_img[y:y+h, x:x+w]
            face_embedding = facenet.embeddings([face_img])[0]
            face_embedding = np.expand_dims(face_embedding, axis=0)

            prediction = model.predict(face_embedding)
            predicted_probabilities = model.predict_proba(face_embedding)
            predicted_name = encoder.inverse_transform(prediction)[0]

            max_probability = np.max(predicted_probabilities)
            max_probability_label = encoder.inverse_transform([np.argmax(predicted_probabilities)])[0]

            threshold = 0.7

            if max_probability >= threshold:
                label = predicted_name
            else:
                label = "Unknown"

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, jpeg = cv.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv.destroyAllWindows()

@app.route('/')
def index():
    video_feed_url = "/video_feed"  # Replace this with your actual video feed URL if needed
    return render_template('index.html', video_feed_url=video_feed_url)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
