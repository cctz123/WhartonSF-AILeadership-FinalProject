import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# ====== Load Models ======
face_model = load_model("models/improved_face_emotion_model.h5")  # Ensure this is trained on FER2013
face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ====== Initialize Mediapipe ======
mp_face_detection = mp.solutions.face_detection

# ====== Facial Emotion Detection Function ======
def detect_facial_emotion(detector, frame):
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        face = frame[y:y+h, x:x+w]
        if face.size == 0: return "No Face"
        
        # Resize and preprocess
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) / 255.0
        face = face.reshape(1, 48, 48, 1)
        pred = face_model.predict(face, verbose=0)
        return face_emotions[np.argmax(pred)]
    return "No Face"

# ====== Main Camera Loop ======
def main():
    print("===== FACIAL EMOTION DETECTION =====")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Larger resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Detection", 960, 540)

    latest_facial_emotion = "Loading..."

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            emotion = detect_facial_emotion(detector, frame)
            if emotion: latest_facial_emotion = emotion

            # Annotate
            overlay = frame.copy()
            cv2.putText(overlay, f"Facial Emotion: {latest_facial_emotion}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Optional FPS display
            fps = 1 / (time.time() - start_time)
            cv2.putText(overlay, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show window
            cv2.imshow("Emotion Detection", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
