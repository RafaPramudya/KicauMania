import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe Library Import
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils

# Model Options
hand_model_options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands = 2,
    running_mode=VisionRunningMode.VIDEO   
)

# CV2 Initialization
cap = cv2.VideoCapture(0)
start_time = time.time()
last_time = start_time

# Main Function
with HandLandmarker.create_from_options(hand_model_options) as model:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # BGR to RGB Convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)

        # Calculate time
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)

        result = model.detect_for_video(image, timestamp_ms)

        # print(result)
        for i in range(len(result.hand_landmarks)):
            hand_landmark = result.hand_landmarks[i]
            drawing_utils.draw_landmarks(frame, hand_landmark, HandLandmarksConnections.HAND_CONNECTIONS)
        
        cv2.imshow("Test", frame)
        
        key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q') : break

cap.release()
cv2.destroyAllWindows()