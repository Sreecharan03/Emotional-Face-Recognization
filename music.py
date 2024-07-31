import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model # type: ignore
import webbrowser

# Load model and labels
try:
    model = load_model("model.h5")
    labels = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()  # Stop execution if model loading fails

# Initialize MediaPipe
holistic = mp.solutions.holistic.Holistic()
hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = True

def reset_emotion():
    np.save("emotion.npy", np.array([""]))

# Load emotion if available
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        
        # Flip the frame horizontally
        frm = cv2.flip(frm, 1)

        # Process the frame with MediaPipe
        results = holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        features = []

        # Process face landmarks
        if results.face_landmarks:
            for i in results.face_landmarks.landmark:
                features.append(i.x - results.face_landmarks.landmark[1].x)
                features.append(i.y - results.face_landmarks.landmark[1].y)

            # Process left hand landmarks
            if results.left_hand_landmarks:
                for i in results.left_hand_landmarks.landmark:
                    features.append(i.x - results.left_hand_landmarks.landmark[8].x)
                    features.append(i.y - results.left_hand_landmarks.landmark[8].y)
            else:
                features.extend([0.0] * 42)

            # Process right hand landmarks
            if results.right_hand_landmarks:
                for i in results.right_hand_landmarks.landmark:
                    features.append(i.x - results.right_hand_landmarks.landmark[8].x)
                    features.append(i.y - results.right_hand_landmarks.landmark[8].y)
            else:
                features.extend([0.0] * 42)

            features = np.array(features).reshape(1, -1)

            # Predict emotion
            prediction = labels[np.argmax(model.predict(features))]

            # Draw prediction on the frame
            cv2.putText(frm, prediction, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion.npy", np.array([prediction]))

        # Draw MediaPipe landmarks on the frame
        if results.face_landmarks:
            drawing.draw_landmarks(frm, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                    connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        if results.left_hand_landmarks:
            drawing.draw_landmarks(frm, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            drawing.draw_landmarks(frm, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User input for language
language = st.text_input("Language")

# Start WebRTC if language is provided and the session is running
if language and st.session_state["run"]:
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button to recommend songs
if st.button("Recommend me songs"):
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        search_query = f"{language} {emotion} song"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        reset_emotion()
        st.session_state["run"] = False
