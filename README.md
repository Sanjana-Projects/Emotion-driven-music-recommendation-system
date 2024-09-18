# Emotion-driven-music-recommendation-system
The Emotion-Driven Music Recommendation System is a dynamic application ulti-task Cascaded Convolutional Networks) for accurate face detection in real-time, ensuring that user emotions are captured efficiently from the video feed.  Once the emotion is identified the system the system recommends songs that match the user’s mood. 

# Emotion-Based Music Recommender System

## Overview

This project is an **AI-powered music recommendation system** that suggests songs based on the user’s detected emotions. By using a combination of **facial emotion recognition** and **Spotify's API**, the system identifies the user’s emotional state and recommends a personalized playlist. The interface is built using **Streamlit** for ease of interaction, and the backend relies on deep learning models for emotion detection.

## Features

- **Real-Time Emotion Detection**: The system uses the webcam to capture real-time facial expressions and classify emotions (e.g., happy, sad, angry).
- **Music Recommendation**: Based on the detected emotion, the system suggests songs from **Spotify** or **YouTube**.
- **Language and Artist Selection**: Users can choose their preferred language and favorite artists for customized playlists.
- **Clickable Image Interface**: Users can interact with clickable song images to choose which song to play.
- **Deep Learning Models**: Utilizes models like **CNN, MTCNN**, and **DCNN** for emotion recognition.
- **API Integration**: Spotify’s Web API is integrated to fetch song data and playlists.

## Installation

1. Use this code

2. **Install Dependencies**:
   Make sure you have **Python 3.7+** installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Pre-Trained Models**:
   Download the necessary pre-trained models for facial emotion recognition (if not included in the repo) and place them in the `models/` folder.

4. **Spotify API Setup**:
   - Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/login) and create a new app.
   - Add your credentials to a `.env` file or configure them within the app as follows:
   ```
   SPOTIFY_CLIENT_ID=your-client-id
   SPOTIFY_CLIENT_SECRET=your-client-secret
   SPOTIFY_REDIRECT_URI=your-redirect-uri
   ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
2. **Access the App**:
   Open your browser and go to `http://localhost:8501` to access the application.
3. **Emotion Detection & Music Recommendation**:
   - Allow access to your webcam.
   - The system will detect your emotion and prompt you to select preferences like language and artist.
   - A list of clickable song images will be displayed, allowing you to play songs via YouTube or Spotify.

## Models Used

- **Convolutional Neural Network (CNN)**: Used for detecting facial emotions in real time.
- **Multitask Cascaded Convolutional Neural Networks (MTCNN)**: For facial feature extraction and recognition.
- **Deep Convolutional Neural Network (DCNN)**: For processing more complex facial cues and improving emotion classification accuracy.

## Dataset

- **FER-2013 Dataset**: This dataset is used to train the emotion detection models. It contains labeled facial images categorized into different emotional states such as happy, sad, angry, etc.

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **Keras & TensorFlow**: For implementing the CNN and DCNN models.
- **OpenCV**: For image and video processing (webcam feed).
- **Spotify Web API**: For fetching and playing songs based on the user’s detected mood.
- **Python**: Core programming language for building the app.

## API Keys

- **Spotify API**: Ensure that you have valid Spotify credentials by registering at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
- **YouTube API (Optional)**: If integrating YouTube for song playback, you can use the YouTube API for advanced features.

## Future Improvements

- **Multilingual Support**: Add more languages for better customization.
- **Mobile Compatibility**: Adapt the system for mobile devices.
- **Additional Emotion Categories**: Expand the range of detectable emotions for a more nuanced music recommendation system.

## Acknowledgments

- **FER-2013 Dataset** for facial emotion recognition.
- **Spotify Web API** for music streaming functionality.
- **Streamlit Community** for providing the platform for building easy-to-use web apps.



This README file covers all aspects of this project, from installation to usage, technologies.
