import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# === 1. Load and preprocess FER-2013 data ===
print("Loading FER-2013 dataset...")
data = pd.read_csv('pretrained_models/fer2013.csv')
pixels = data['pixels'].tolist()
faces = np.array([np.fromstring(p, sep=' ') for p in pixels])
faces = faces.reshape(-1, 48, 48, 1)
faces = np.repeat(faces, 3, axis=-1)  # Convert to 3 channels
faces = faces / 255.0  # Normalize
faces = preprocess_input(faces)  # ResNet50 preprocessing

emotions = to_categorical(data['emotion'].values, num_classes=7)
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# === 2. Augmentation ===
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
datagen.fit(X_train)

# === 3. Build Fine-tuned ResNet50 model ===
def build_model():
    input_tensor = Input(shape=(48, 48, 3))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

print("Building and compiling model...")
model = build_model()
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# === 4. Train Model ===
print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val)
)

# === 5. Save model ===
os.makedirs("models", exist_ok=True)
model.save("models/improved_face_emotion_model.h5")
print("Model saved to models/improved_face_emotion_model.h5")

# === 6. Evaluation ===
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print(classification_report(y_true, y_pred_classes, target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']))

# === 7. Visualization ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("models/improved_training_plots.png")
plt.show()
