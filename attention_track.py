import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# Input the student name
student_name = input("Enter the student's name: ")
csv_file = f"{student_name}_attention_data.csv"

# Create or overwrite the CSV file and write the header
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Attention State", "Duration (seconds)"])

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Coordinates for left and right eyes (adjusted for landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7, 163, 144]
RIGHT_EYE = [362, 385, 386, 263, 380, 373, 374, 374, 380, 362]

# Start capturing video
cap = cv2.VideoCapture(0)  # Use webcam

# Function to get the eye position
def get_eye_position(landmarks, eye_indices):
    eye_points = []
    for idx in eye_indices:
        x = landmarks[idx].x
        y = landmarks[idx].y
        eye_points.append((x, y))
    return np.mean(eye_points, axis=0)

# Function for blink detection
def detect_blink(landmarks, eye_indices):
    top_point = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    bottom_point = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    distance = np.linalg.norm(top_point - bottom_point)
    return distance

# Function for head pose estimation
def detect_head_pose(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]

    eye_distance = abs(left_eye.x - right_eye.x)
    if eye_distance > 0.05:  # Threshold for head turn detection
        if left_eye.x < right_eye.x and nose_tip.y < left_eye.y:
            return 1  # Head turned left
        elif left_eye.x > right_eye.x and nose_tip.y < right_eye.y:
            return -1  # Head turned right
    return 0  # Head facing forward

# Variables for state tracking and timer
previous_state = None
state_start_time = time.time()
state_durations = {"Focused": 0, "Neutral": 0, "Distracted": 0}

# Function to update state duration
def update_state_duration(current_state):
    global previous_state, state_start_time, state_durations
    current_time = time.time()
    if previous_state is not None:
        elapsed_time = current_time - state_start_time
        state_durations[previous_state] += elapsed_time
    state_start_time = current_time
    previous_state = current_state

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face landmarks
    results = face_mesh.process(rgb_frame)

    # Initialize attention scores
    gaze_score = 0
    blink_score = 0
    head_pose_score = 0
    attention_status = "Neutral"

    # Draw face landmarks and compute attention scores
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Gaze Detection
            left_eye_position = get_eye_position(face_landmarks.landmark, LEFT_EYE)
            right_eye_position = get_eye_position(face_landmarks.landmark, RIGHT_EYE)

            if left_eye_position[0] < 0.7 and right_eye_position[0] < 0.7:
                gaze_score = 1  # Looking at the screen
            else:
                gaze_score = 0  # Looking away

            # Blink Detection
            left_blink = detect_blink(face_landmarks.landmark, LEFT_EYE)
            right_blink = detect_blink(face_landmarks.landmark, RIGHT_EYE)

            if left_blink < 0.02 or right_blink < 0.02:
                blink_score = 1  # Blink detected

            # Head Pose Estimation
            head_pose_score = detect_head_pose(face_landmarks.landmark)
    
    # Calculate overall attention score
    total_score = 0
    total_score += gaze_score * 0.5
    total_score -= head_pose_score * 0.1
    total_score += blink_score * 0.2

    # Determine attention status
    if total_score >= 0.6:
        attention_status = "Focused"
    elif total_score > 0.4:
        attention_status = "Neutral"
    else:
        attention_status = "Distracted"

    # Update the timer for the current state
    update_state_duration(attention_status)

    # Display attention status and timer on screen
    cv2.putText(frame, f"Attention: {attention_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Focused: {state_durations['Focused']:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Neutral: {state_durations['Neutral']:.1f}s", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Distracted: {state_durations['Distracted']:.1f}s", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Attention Tracker", frame)

    # Log the durations to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([attention_status, f"{state_durations[attention_status]:.2f}"])

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
