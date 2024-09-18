import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser
import os
from mtcnn import MTCNN  # Import MTCNN
from spotify_handler import get_spotify_track_url
import pandas as pd

# Paths to static files
MODEL_PATH = "emotion_detection_model.keras"
LABELS_PATH = "labels.npy"
DATA_CSV_PATH = "emotion_data.csv"

# Function to save data to CSV
def save_data(emotion, music_recommendation):
    data = {
        'Date': [pd.Timestamp.now()],
        'Emotion': [emotion],
        'MusicRecommendation': [music_recommendation]
    }
    df = pd.DataFrame(data)
    
    if os.path.exists(DATA_CSV_PATH):
        df.to_csv(DATA_CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_CSV_PATH, index=False)

# Emotion Processor Class
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, model, emotion_labels):
        self.model = model
        self.emotion_labels = emotion_labels
        self.emotion = None
        self.detector = MTCNN()  # Initialize MTCNN detector

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        faces = self.detector.detect_faces(frm)  # Detect faces with MTCNN

        for face in faces:
            x, y, w, h = face['box']
            face_region = frm[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = gray_face.reshape(1, 48, 48, 1) / 255.0

            pred = self.model.predict(gray_face)
            self.emotion = self.emotion_labels[np.argmax(pred)]
            cv2.putText(frm, self.emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frm, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save detected emotion if it's detected
        if self.emotion:
            np.save("emotion.npy", np.array([self.emotion]))

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Underline the title
st.markdown("<h1 style='text-align: left; text-decoration: underline;'>Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)

st.write("The system will detect your emotion and offer options to play music based on the detected emotion:")

# Custom styling for input labels
st.markdown("<h3 style='font-size:30px; margin-bottom: 0px;'>Music Preferences</h3>", unsafe_allow_html=True)

# Increase font size for input text
st.markdown("<p style='font-size:18px; margin-bottom: 0px;'>Enter language:</p>", unsafe_allow_html=True)
lang = st.text_input("", key="lang_input")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='font-size:18px;'>Enter name of singer:</p>", unsafe_allow_html=True)
singer = st.text_input("", key="singer_input")

# Load model and labels
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at {MODEL_PATH}")

if os.path.exists(LABELS_PATH):
    try:
        emotion_labels = np.load(LABELS_PATH)
    except Exception as e:
        st.error(f"Error loading labels file: {e}")
else:
    st.error(f"Labels file not found at {LABELS_PATH}")

# Streamlit video capture
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

    def toggle_streaming():
        st.session_state.streaming = not st.session_state.streaming

    if st.button("Start/Stop Webcam"):
        toggle_streaming()

    if st.session_state.streaming:
        webrtc_streamer(
            key="emotion",
            video_processor_factory=lambda: EmotionProcessor(model, emotion_labels)
        )
    else:
        # No direct stop option available, this will just prevent the webcam from being processed
        webrtc_streamer(
            key="emotion",
            video_processor_factory=lambda: EmotionProcessor(model, emotion_labels),
            desired_playing_state=False
        )

# Clickable images for music options
if os.path.exists("emotion.npy") and os.path.getsize("emotion.npy") != 0:
    emotion = np.load("emotion.npy")[0]
    query = f"{emotion} song by {singer} in {lang}"

    youtube_query = query.replace(' ', '+')
    youtube_url = f"https://www.youtube.com/results?search_query={youtube_query}"
    spotify_track_url = get_spotify_track_url(emotion, lang, singer)

    # Save data to CSV
    save_data(emotion, spotify_track_url if spotify_track_url else youtube_url)

    st.subheader("Choose a Platform to play your Song:")

    col1, col2 = st.columns(2)

    with col1:
        youtube_image = "youtube.jpg"  # Replace with the path to your YouTube image
        if os.path.exists(youtube_image):
            st.image(youtube_image, use_column_width=True)
            if st.button("Play on YouTube", key="youtube_button"):
                webbrowser.open(youtube_url)
        else:
            st.warning("YouTube image not found!")

    with col2:
        spotify_image = "spotify.jpg"  # Replace with the path to your Spotify image
        if os.path.exists(spotify_image):
            st.image(spotify_image, use_column_width=True)
            if st.button("Play on Spotify", key="spotify_button"):
                webbrowser.open(spotify_track_url)
        else:
            st.warning("Spotify image not found!")
else:
    st.warning("Please let me capture your emotion first.")
