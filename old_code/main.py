import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import mediapipe as mp
from keras.models import load_model
from speechbrain.inference import EncoderClassifier
from keras.applications.resnet50 import preprocess_input 
import time


# Load models
face_model = load_model("models/face_emotion_model.h5")
classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")

face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection

# Shared state
latest_facial_emotion = "Loading..."
latest_speech_emotion = "Loading..."
running = True

def detect_facial_emotion(detector, frame):
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            return "No Face"

        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32) / 255.0  # ✅ Normalize
        face = face.reshape(1, 48, 48, 3)        # ✅ 3 channels expected

        pred = face_model.predict(face, verbose=0)
        return face_emotions[np.argmax(pred)]
    return "No Face"


def detect_speech_emotion_loop():
    global latest_speech_emotion, running
    while running:
        try:
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, 16000)
                pred = classifier.classify_file(f.name)
                latest_speech_emotion = pred[3]
        except Exception as e:
            latest_speech_emotion = f"Error: {e}"

def main():
    global latest_facial_emotion, running

    print("===== EMOTION DETECTION SYSTEM =====")
    print("Press 'q' to quit.")
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

    # Start speech thread
    threading.Thread(target=detect_speech_emotion_loop, daemon=True).start()

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            emotion = detect_facial_emotion(detector, frame)
            if emotion: latest_facial_emotion = emotion

            overlay = frame.copy()
            cv2.putText(overlay, f"Facial: {latest_facial_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, f"Speech: {latest_speech_emotion}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
