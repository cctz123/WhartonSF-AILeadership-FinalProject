import cv2
import numpy as np
import threading
import mediapipe as mp
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.applications.resnet50 import preprocess_input

# Load emotion labels
face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection

# Shared state
latest_facial_emotion = "Loading..."
running = True

# Build ResNet50-based model for emotion detection
def build_resnet50_model():
    input_tensor = Input(shape=(48, 48, 3))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

face_model = build_resnet50_model()

# Dummy predict call to initialize weights (optional but safe)
face_model.predict(np.zeros((1, 48, 48, 3)), verbose=0)

def preprocess_face(face):
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC)
    if len(face.shape) == 2 or face.shape[2] == 1:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.reshape(face, (1, 48, 48, 3))
    return face

def detect_facial_emotion(frame):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                return "No Face"
            face = preprocess_face(face)
            pred = face_model.predict(face, verbose=0)
            return face_emotions[np.argmax(pred)]
        return "No Face"

def main():
    global latest_facial_emotion, running

    print("===== EMOTION DETECTION SYSTEM =====")
    print("Press 'q' to quit.")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            emotion = detect_facial_emotion(frame)
            if emotion:
                latest_facial_emotion = emotion

            overlay = frame.copy()
            cv2.putText(overlay, f"Facial: {latest_facial_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Emotion Detection", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
https://www.kaggle.com/datasets/deadskull7/fer2013
https://blog.bytescrum.com/create-a-real-time-face-emotion-detector-with-python-and-deep-learning#heading-3-dataset-fer-2013
'''
