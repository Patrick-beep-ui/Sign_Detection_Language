import cv2
import numpy as np
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import time

# MediaPipe values
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Detect with Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
    image.flags.writeable = False  # Prevent modifications during processing
    results = model.process(image)  # Perform prediction
    image.flags.writeable = True  # Allow modifications again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    return image, results

# Draw landmarks on image
def draw_landmarks(image, results):
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 250, 110), thickness=1, circle_radius=1))
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw left and right hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Extract Key Points
def get_key_points(results):
    # Check if landmarks exist, add error handling
    if results.pose_landmarks:
        pose_lm = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose_lm = np.zeros(33*4)  # Pose landmarks have 33 points, and each point has 4 values (x, y, z, visibility)
        
    if results.face_landmarks:
        face_lm = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face_lm = np.zeros(468*3)  # Face landmarks for 1400 values

    if results.left_hand_landmarks:
        lh_lm = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh_lm = np.zeros(21*3)  # Left hand has 21 points, 3 values per point (x, y, z)
        
    if results.right_hand_landmarks:
        rh_lm = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh_lm = np.zeros(21*3)  # Right hand has 21 points, 3 values per point (x, y, z)
    
    return np.concatenate([pose_lm, face_lm, lh_lm, rh_lm]) 

# Start video capture
cap = cv2.VideoCapture(0)
# Set and access the Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        
        # Perform Mediapipe detection on the frame
        image, results = mediapipe_detection(frame, holistic)
        #print(results)  # Print results for debugging
        
        # Draw landmarks on the image
        draw_landmarks(image, results)
        
        # Extract key points
        get_key_points(results)
        
        # Flip the frame horizontally to remove mirror effect
        flipped_frame = cv2.flip(image, 1)  # Use the processed image with the landmarks
        
        # Display the frame with landmarks
        cv2.imshow('Sign Detection', flipped_frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# SETUP FOLDER FOR COLLECTION
DATA_PATH = os.path.join('MP_Data') # Path for exported data
actions = np.array(['hello', 'thanks', 'iloveyou']) # Actions to detect
no_sequences = 30 # 30 videos worth of data
sequence_length = 30 # 30 frames length for videos

for action in actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: 
            pass