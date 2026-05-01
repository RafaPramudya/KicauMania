from contextlib import ExitStack

import cv2
import cv2.typing as cvt
import mediapipe as mp
import numpy as np
import time

import csv
import yaml
import argparse

from utils import (
    visualize_face_detection,
    visualize_face_landmarker,
    visualize_hand_detection,
    max_normalize,
    normalize_with_base
)

# Model Path
HAND_LANDMARKER_MODEL_PATH = "model/hand_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = "model/face_landmarker.task"
FACE_DETECTOR_MODEL_PATH   = "model/blazeface_short.tflite"

# CLI argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--yml', required=True)
parser.add_argument('--samples', default=100, type=int)
parser.add_argument('--delay', default=.1, type=float)
parser.add_argument('--overwrite', action='store_true')

args = parser.parse_args()

num_samples = args.samples
delay       = args.delay
overwrite   = args.overwrite

# YAML metadata read
with open(args.yml, "r") as f:
    yaml_data = yaml.load(f, Loader=yaml.FullLoader)

model_name          = yaml_data["name"]
model_path          = yaml_data["path"]
model_training_data = yaml_data["training_data"]
min_face_req        = yaml_data["min_face"]
min_hand_req        = yaml_data["min_hand"]
face_detection_type = yaml_data["face_detection_type"]
labels              = yaml_data["labels"]
features            = yaml_data["features"]
normalization       = yaml_data["normalization"]

use_hand_detector   = (min_hand_req > 0)
use_face_detector   = (min_face_req > 0 and face_detection_type == "Detection")
use_face_landmarker = (min_face_req > 0 and face_detection_type == "Landmarker")

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
    running_mode = VisionRunningMode.VIDEO,
    num_hands = min(1, min_hand_req)
) if use_hand_detector else None

face_D_model_options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH),
    running_mode = VisionRunningMode.VIDEO
) if use_face_detector else None

face_L_model_options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = min(1, min_face_req)
) if use_face_landmarker else None

# CV2 Initialization
cap = cv2.VideoCapture(0)
start_time = time.time()
last_time = start_time

# Main Function
with ExitStack() as stack, open(model_training_data, 'w' if overwrite else 'a', newline='') as f:

    # Model stack initialization
    hand_model = stack.enter_context(
        HandLandmarker.create_from_options(hand_model_options)
    ) if use_hand_detector else None
    face_D_model = stack.enter_context(
        FaceDetector.create_from_options(face_D_model_options)
    ) if use_face_detector else None
    face_L_model = stack.enter_context(
        FaceLandmarker.create_from_options(face_L_model_options)
    ) if use_face_landmarker else None

    csv_writer = csv.writer(f)
    recording = False
    
    num_samples_recorded = 0
    num_labels_recorded = 0

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret: break

        # BGR to RGB Convert
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)

        # Calculate time
        current_time = time.time()
        timestamp_ms = int((current_time - start_time) * 1000)

        if num_samples_recorded >= num_samples:
            recording = False
            num_samples_recorded = 0
            num_labels_recorded += 1
        if num_labels_recorded >= len(labels): break

        # This variable will be false if one of the detector does'nt meet the required minimum
        valid_record = True

        if use_hand_detector:
            hand_result = hand_model.detect_for_video(image, timestamp_ms)
            frame = visualize_hand_detection(frame, hand_result)
            if len(hand_result.hand_landmarks) < min_hand_req: valid_record = False
        
        if use_face_detector:
            face_D_result = face_D_model.detect_for_video(image, timestamp_ms)
            frame = visualize_face_detection(frame, face_D_result)
            if len(face_D_result.detections) < min_face_req : valid_record = False
        
        if use_face_landmarker:
            face_L_result = face_L_model.detect_for_video(image, timestamp_ms)
            frame = visualize_face_landmarker(frame, face_L_result)
            if len(face_L_result.face_landmarks) < min_face_req : valid_record = False

        if valid_record and recording:
            coords = []
            coords.extend(
                [lm.x, lm.y, lm.z] 
                for hand_landmark in (hand_result.hand_landmarks if use_hand_detector else [[]])
                for lm in hand_landmark
            )
            coords.extend(
                [kp.x, kp.y, 0.0] 
                for face_detection in (face_D_result.detections[:min_face_req] if use_face_detector else [type('obj', (object,), {'keypoints': []})()])
                for kp in face_detection.keypoints
            )
            coords.extend(
                [lm.x, lm.y, lm.z] 
                for face_landmark in (face_L_result.face_landmarks if use_face_landmarker else [[]])
                for lm in face_landmark
            )
            coords = np.array(coords)

            if normalization == "max_norm":
                coords = max_normalize(coords)
            elif normalization == "palm_base":
                if use_hand_detector:
                    coords = normalize_with_base(coords, [hand_result.hand_landmarks[0][0].x, hand_result.hand_landmarks[0][0].y, hand_result.hand_landmarks[0][0].z])
            elif normalization == "nose_base":
                if use_face_landmarker:
                    coords = normalize_with_base(coords, [face_L_result.face_landmarks[0][4].x, face_L_result.face_landmarks[0][4].y, face_L_result.face_landmarks[0][4].z])
                elif use_face_detector:
                    coords = normalize_with_base(coords, [face_D_result.detections[0].keypoints[2].x, face_D_result.detections[0].keypoints[2].y, 0.0])
            
            feature = []
            for ft in features:
                if ft == "hand_face_distance":
                    if use_face_landmarker:
                        feature.extend([
                            abs(hand_result.hand_landmarks[0][0].x - face_L_result.face_landmarks[0][4].x),
                            abs(hand_result.hand_landmarks[0][0].y - face_L_result.face_landmarks[0][4].y),
                            abs(hand_result.hand_landmarks[0][0].z - face_L_result.face_landmarks[0][4].z)
                        ])
                    elif use_face_detector:
                        feature.extend([
                            abs(hand_result.hand_landmarks[0][0].x - face_D_result.detections[0].keypoints[2].x),
                            abs(hand_result.hand_landmarks[0][0].y - face_D_result.detections[0].keypoints[2].y),
                            abs(hand_result.hand_landmarks[0][0].z - 0.0)
                        ])
            
            coords  = np.array(coords)
            feature = np.array(feature)

            csv_writer.writerow([num_labels_recorded] + coords.tolist() + feature.tolist())
            num_samples_recorded += 1

        if recording:   helper_text = f"Jumlah sampel terekam {num_samples_recorded} / {num_samples}"
        else:           helper_text = f"Tekan 'P' untuk memulai merekam {labels[num_labels_recorded]}"

        cv2.putText(frame, helper_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        cv2.imshow("Perekaman Sampel", frame)
        
        key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q') : break
        if key == ord('p') or key == ord('P') : recording = True
        if key == ord('s') or key == ord('S') : num_labels_recorded += 1

cap.release()
cv2.destroyAllWindows()