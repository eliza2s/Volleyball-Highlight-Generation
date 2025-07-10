import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing utilities to visualize pose
mp_drawing = mp.solutions.drawing_utils

# Video Source (0 for webcam, or replace with video file path)
cap = cv2.VideoCapture(r"C:\Users\Eliza\OneDrive\Desktop\Mediapipe\venv\nebraska-volleyball.mp4")

# Action Detection Functions
def is_spike(landmarks, frame_height):
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame_height
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height

    return (left_wrist_y < shoulder_y - 50) or (right_wrist_y < shoulder_y - 50)

def is_block(landmarks, frame_height):
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame_height
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height

    return (left_wrist_y < shoulder_y and right_wrist_y < shoulder_y)

# Main Loop
frame_count = 0
highlight_log = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame or video ended.")
        break

    frame_count += 1
    print(f"\nProcessing frame {frame_count}...")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        print("Pose detected!")

        # Draw landmarks on the frame for visualization
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        frame_height = frame.shape[0]

        # Debug positions
        lw_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height
        rw_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame_height
        sh_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height
        print(f"Hands (L/R): {lw_y:.2f}, {rw_y:.2f} | Shoulder: {sh_y:.2f}")

        if is_spike(landmarks, frame_height):
            print("SPIKE DETECTED!")
            highlight_log.append({"action": "SPIKE", "frame": frame_count})

        elif is_block(landmarks, frame_height):
            print("BLOCK DETECTED!")
            highlight_log.append({"action": "BLOCK", "frame": frame_count})
    else:
        print("No pose landmarks detected.")

    # Show frame with landmarks
    cv2.imshow("MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save highlights to JSON
with open("highlights.json", "w") as f:
    json.dump(highlight_log, f, indent=2)

print("\nHighlights saved to highlights.json!")
