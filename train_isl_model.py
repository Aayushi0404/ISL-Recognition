import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATASET_DIR   = r"C:\Users\vaayu\Downloads\ISL PROJECT\ISL_skeleton_data"
MODEL_SAVE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isl_model.h5")

IMG_SIZE      = 64
NUM_CLASSES   = 26
VALID_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letter_to_idx = {letter: idx for idx, letter in enumerate(VALID_LETTERS)}

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
print("Loading dataset...")
images = []
labels = []

for folder_name in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    if not folder_name.isalpha() or len(folder_name) != 1:
        print(f"  Skipping: '{folder_name}'")
        continue

    letter = folder_name.upper()
    if letter not in letter_to_idx:
        continue

    idx   = letter_to_idx[letter]
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg','.jpeg','.png'))]

    # Limit to 500 images per letter to prevent imbalance and overfitting
    files = files[:500]
    print(f"  Loading {letter} (class {idx}): {len(files)} images")

    for fname in files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(idx)

if len(images) == 0:
    print("No images loaded! Check DATASET_DIR path.")
    exit(1)

images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels)

print(f"\nTotal images  : {len(images)}")
print(f"Image shape   : {images[0].shape}")
print(f"Classes found : {len(set(labels))}")

# ─── SPLIT ───────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")

# ─── AUGMENTATION ────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ─── MODEL ───────────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = Sequential([
    Conv2D(32,  (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(64,  (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ─── CALLBACKS ───────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=0.00001, verbose=1)
]

# ─── TRAIN ───────────────────────────────────────────────────────────────────
print("\nTraining started... (10-20 minutes)")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# ─── EVALUATE ────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nFinal Test Accuracy : {acc*100:.2f}%")
print(f"Final Test Loss     : {loss:.4f}")
print(f"Model saved to      : {MODEL_SAVE}")

# ─── PLOT ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss'); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_graph.png'))
plt.show()
print("Training graph saved!")
