import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

try:
    model = load_model('model.h5', compile=False)
    print("Model loaded successfully")
    print("Model input shape:", model.input_shape)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Update after confirming class_indices
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_classifier.empty():
    print("Error: Could not load haarcascade_frontalface_default.xml")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

start_time = time.time()
print("Starting video capture. Press 'q' to quit.")
time.sleep(2)  # Allow camera to warm up
while True:
    if time.time() - start_time > 60:  # Run for 5 minutes
        print("60 seconds elapsed. Exiting.")
        break
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = img_to_array(roi_gray)
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            # Removed: roi = roi / 255.0

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = np.max(prediction) * 100
            print(f"Raw predictions: {list(zip(emotion_labels, prediction*100))}")
            label_text = f"{label} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing face: {e}")
            continue

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended")
#To install again all modules type: pip install -r requirements.txt
#TO run the script type: python emotion_recognition.py
