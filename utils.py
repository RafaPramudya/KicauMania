import math
import numpy as np
import numpy.typing as npt
import cv2
import cv2.typing as cvt
import mediapipe as mp
from typing import Union, Tuple

HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
FaceLandmarkerConnections = mp.tasks.vision.FaceLandmarksConnections
drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles

def _norm2pixel_(normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    def _isvalidnorm_(val: float) -> bool: return (val > 0 or math.isclose(0, val)) and (val < 1 or math.isclose(1, val))

    if not (_isvalidnorm_(normalized_x) and _isvalidnorm_(normalized_y)): return None

    x = min(math.floor(normalized_x * image_width),  image_width - 1)
    y = min(math.floor(normalized_y * image_height), image_height - 1)
    return x, y

def _normalize_(arr: npt.NDArray[np.float64]) -> np.ndarray:
    max_val = np.max(np.abs(arr))
    arr = arr / max_val

    return arr
def max_normalize(coords: npt.NDArray[np.float64]) -> np.ndarray:
    coords = coords.reshape(-1, 3)

    coords = _normalize_(coords)
    return coords.flatten()

def normalize_with_base(coords: npt.NDArray[np.float64], base: np.ndarray) -> np.ndarray:
    coords = coords.reshape(-1, 3)
    coords = coords - base

    coords = _normalize_(coords)
    return coords.flatten()


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

        category        = detection.categories[0]
        category_name   = category.category_name
        category_name   = '' if category_name is None else category_name
        probality       = round(category.score, 2)
        # result_text = f"{category_name} ({str(probality)})"
        result_text     = ""        
        text_location   = (10 + bbox.origin_x, 20 + bbox.origin_y)
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