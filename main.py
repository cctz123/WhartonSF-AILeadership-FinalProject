import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import mediapipe as mp
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import ResNet50
from speechbrain.inference import EncoderClassifier

# Load models
face_model = load_model("models/face_emotion_model.h5")
classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")

face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Shared state
latest_facial_emotion = "Loading..."
latest_speech_emotion = "Loading..."
running = True

# Function to use ResNet50 for feature extraction and emotion detection
def detect_facial_emotion(frame):
    # Load ResNet50 with weights pre-trained on ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    
    # Freeze the ResNet50 model layers
    base_model.trainable = False
    
    # Add custom classification layers on top of ResNet50
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax')(x)  # 7 emotions as output
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=x)

    # Check for face detection and process face image
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face = frame[y:y+h, x:x+w]
            if face.size == 0: return "No Face"
            face = cv2.resize(face, (48, 48))

            # Convert grayscale to RGB
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Preprocess the face image for ResNet50
            face_rgb = np.reshape(face_rgb, (1, 48, 48, 3))  # Add batch dimension

            # Make the prediction
            pred = model.predict(face_rgb, verbose=0)
            return face_emotions[np.argmax(pred)]
        return "No Face"

# Function to detect speech emotion continuously
def detect_speech_emotion_loop():
    global latest_speech_emotion, running
    while running:
        try:
            # Record 3 seconds of audio
            audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
            sd.wait()

            # Save the audio and classify it
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, 16000)
                print(f"[✅] Audio saved to {f.name}")
                pred = classifier.classify_file(f.name)
                latest_speech_emotion = pred[3]
        except Exception as e:
            latest_speech_emotion = f"Error: {e}"

# Main function to handle video capture and emotion detection
def main():
    global latest_facial_emotion, running

    print("===== EMOTION DETECTION SYSTEM =====")
    print("Starting camera and microphone...")
    print("Press 'q' to quit.")

    # Initialize webcam and start speech emotion thread
    cap = cv2.VideoCapture(1)
    threading.Thread(target=detect_speech_emotion_loop, daemon=True).start()

    # Initialize Mediapipe face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Detect facial emotion
            facial_emotion = detect_facial_emotion(frame)
            if facial_emotion: latest_facial_emotion = facial_emotion

            # Overlay both facial and speech emotions on the frame
            overlay = frame.copy()
            cv2.putText(overlay, f"Facial: {latest_facial_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, f"Speech: {latest_speech_emotion}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Emotion Detection", overlay)

            # Keypress to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
