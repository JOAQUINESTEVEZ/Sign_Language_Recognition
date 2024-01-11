import cv2
import streamlit as st
import mediapipe as mp
import keyboard
import numpy as np
from tensorflow.keras.models import load_model  # Import load_model function
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load your pre-trained model
model = load_model("actionv2.h5")

# Actions
actions = np.array(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
)

# Streamlit configuration
st.set_page_config(page_title="ASL Learning", page_icon="✋", layout="wide")

# New detection variables
sequence = []
sentence = [' ']
threshold = 0.7
buffer_size = 30
change_detected = False
frames_after_change = 0
frames_to_skip_after_change = 5

# Set up the camera
cap = cv2.VideoCapture(0)

# Set up mediapipe model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set page title
st.title("SignSavvy: Your Gateway to ASL Mastery")

# Streamlit layout
col1, col2 = st.columns(2)

# Display a message or content related to the ongoing practice
col1.header("Let's Practice some letters 🤟")
col1.write("Immerse yourself in the world of ASL. Practice making signs and enhance your skills!")
start_button = col1.button("Start Practice")

if start_button:
    finish_button = col1.button("Finish Practice")

    with col1:
        st.image("alphabet_chart.jpg", use_column_width=True)  # Replace with the actual ASL alphabet image URL

    st.sidebar.header("Practice Area")
    practice_area = st.sidebar.empty()

    # Output box for displaying letters
    letter_output_container = col2.empty()
    key_for_text_area = 1
    letter_output = letter_output_container.text_area("Sign Translation:", value=' '.join(sentence), key=key_for_text_area)

    while not finish_button:
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-buffer_size:]

        if len(sequence) == buffer_size:
            if finish_button:
                finish_button = True  # Finish practice if the button is clicked
            else:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                # Change detection logic
                if res[np.argmax(res)] > threshold:
                    if not change_detected:
                        frames_after_change = 0
                        change_detected = True
                    else:
                        frames_after_change += 1
                        if frames_after_change > frames_to_skip_after_change:
                            # Perform prediction and update sentence
                            predicted_letter = actions[np.argmax(res)]

                            if len(sentence) > 0 and predicted_letter != sentence[-1]:
                                sentence.append(predicted_letter)

                            # Update output box
                            key_for_text_area = key_for_text_area + 1
                            letter_output_container.text_area("Sign Translation:", value=' '.join(sentence), key=key_for_text_area)

                else:
                    change_detected = False

        # Display to practice area
        practice_area.image(image, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()