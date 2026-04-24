from typing import Union, Tuple
import math

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
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils

# Model Options
hand_model_options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands = 2,
    running_mode = VisionRunningMode.VIDEO   
)
face_model_options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path="blazeface_short.tflite"),
    running_mode = VisionRunningMode.VIDEO
)

# Utils function
def _norm2pixel_(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    def _isvalidnorm_(val: float) -> bool: return (val > 0 or math.isclose(0, val)) and (val < 1 or math.isclose(1, val))

    if not (_isvalidnorm_(normalized_x) and _isvalidnorm_(normalized_y)): return None

    x = min(math.floor(normalized_x * image_width),  image_width - 1)
    y = min(math.floor(normalized_y * image_height), image_height - 1)
    return x, y

# def visualize_hand_detection() deprecated, using drawing_utils from MediaPipe
def visualize_face_detection(image, detection_result) -> np.ndarray:
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point   = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

        cv2.rectangle(annotated_image, start_point, end_point, (255, 0, 0), 3)

        for keypoint in detection.keypoints:
            keypoint_px = _norm2pixel_(keypoint.x, keypoint.y, width, height)
            cv2.circle(annotated_image, keypoint_px, 2, (255, 255, 255), 2)

        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probality = round(category.score, 2)
        result_text = f"{category_name} ({str(probality)})"
        text_location = (10 + bbox.origin_x, 20 + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    
    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# CV2 Initialization
cap = cv2.VideoCapture(0)
start_time = time.time()
last_time = start_time

# Main Function
with HandLandmarker.create_from_options(hand_model_options) as hand_model, FaceDetector.create_from_options(face_model_options) as face_model:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # BGR to RGB Convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)

        # Calculate time
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)

        hand_result = hand_model.detect_for_video(image, timestamp_ms)
        face_result = face_model.detect_for_video(image, timestamp_ms)

        # print(hand_result)
        # print(face_result)
        for i in range(len(hand_result.hand_landmarks)):
            hand_landmark = hand_result.hand_landmarks[i]
            drawing_utils.draw_landmarks(frame, hand_landmark, HandLandmarksConnections.HAND_CONNECTIONS)

        frame = visualize_face_detection(frame, face_result)

        cv2.imshow("Test", frame)
        
        key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q') : break

cap.release()
cv2.destroyAllWindows()