import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import zipfile

# Define paths
MODEL_PATH = "emotion_detection_model.keras"  # Changed to .keras
LABELS_PATH = "labels.npy"  # Path to save labels

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title("Emotion Detection Model Training")

uploaded_file = st.file_uploader("Upload a ZIP file with 'train' and 'test' directories", type="zip")

if uploaded_file is not None:
    with open("dataset.zip", "wb") as f:
        f.write(uploaded_file.getvalue())
    st.text("Dataset uploaded successfully.")

    if st.button("Train Model", key="train_model"):
        # Extract dataset
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
            with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("dataset")
        
        # Define ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = train_datagen.flow_from_directory(
            "dataset/train",
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=32,
            class_mode="categorical",
            subset="training"
        )
        validation_generator = train_datagen.flow_from_directory(
            "dataset/train",
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=32,
            class_mode="categorical",
            subset="validation"
        )
        
        # Save class indices to labels.npy
        labels = list(train_generator.class_indices.keys())
        np.save(LABELS_PATH, labels)
        st.text(f"Labels saved to {LABELS_PATH}: {labels}")

        model = create_model()
        checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min')
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=[checkpoint]
        )

        st.success("Model trained successfully!")
