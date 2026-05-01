from typing import Union, Tuple
import math

import cv2
import cv2.typing as cvt
import mediapipe as mp
import numpy as np
import time

# Model Path
HAND_LANDMARKER_MODEL_PATH = "model/hand_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = "model/face_landmarker.task"
FACE_DETECTOR_MODEL_PATH   = "model/blazeface_short.tflite"

# Mediapipe Library Import
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerConnections = mp.tasks.vision.FaceLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles

# Model Options
hand_model_options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
    running_mode = VisionRunningMode.VIDEO   ,
    num_hands = 1
)
face_D_model_options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH),
    running_mode = VisionRunningMode.VIDEO
)
face_L_model_options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
)

# Utils function
def _norm2pixel_(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    def _isvalidnorm_(val: float) -> bool: return (val > 0 or math.isclose(0, val)) and (val < 1 or math.isclose(1, val))

    if not (_isvalidnorm_(normalized_x) and _isvalidnorm_(normalized_y)): return None

    x = min(math.floor(normalized_x * image_width),  image_width - 1)
    y = min(math.floor(normalized_y * image_height), image_height - 1)
    return x, y

def visualize_hand_detection(image: cvt.MatLike, detection_result) -> np.ndarray:
    annotated_image = image.copy()
    for i in range(len(detection_result.hand_landmarks)):
        hand_landmark = detection_result.hand_landmarks[i]
        drawing_utils.draw_landmarks(annotated_image, hand_landmark, HandLandmarksConnections.HAND_CONNECTIONS)

    return annotated_image

def visualize_face_detection(image: cvt.MatLike, detection_result) -> np.ndarray:
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

def visualize_face_landmarker(image: cvt.MatLike, detection_result) -> np.ndarray:
    annotated_image = image.copy()
    for i in range(len(detection_result.face_landmarks)):
        face_landmarks = detection_result.face_landmarks[i]

        drawing_utils.draw_landmarks(
            annotated_image,
            face_landmarks,
            FaceLandmarkerConnections.FACE_LANDMARKS_TESSELATION,
            None, 
            drawing_styles.get_default_face_mesh_tesselation_style()
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            face_landmarks,
            FaceLandmarkerConnections.FACE_LANDMARKS_LEFT_IRIS,
            None,
            drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            face_landmarks,
            FaceLandmarkerConnections.FACE_LANDMARKS_RIGHT_IRIS,
            None,
            drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image

# CV2 Initialization
cap = cv2.VideoCapture(0)
start_time = time.time()
last_time = start_time

# Main Function
with    HandLandmarker.create_from_options(hand_model_options) as hand_model,\
        FaceDetector.create_from_options(face_D_model_options) as face_D_model,\
        FaceLandmarker.create_from_options(face_L_model_options) as face_L_model\
    :
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
        # face_D_result = face_D_model.detect_for_video(image, timestamp_ms)
        face_L_result = face_L_model.detect_for_video(image, timestamp_ms)

        # print(hand_result)
        # print(face_D_result)
        print(face_L_result)

        frame = visualize_hand_detection(frame, hand_result)
        # frame = visualize_face_detection(frame, face_D_result)
        frame = visualize_face_landmarker(frame, face_L_result)

        cv2.imshow("Kicau Mania", frame)
        
        key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q') : break

cap.release()
cv2.destroyAllWindows()