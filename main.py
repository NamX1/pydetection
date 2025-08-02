import numpy as np
import mediapipe as mp
import cv2
from typing import Tuple, Union
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants for visualization
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Red for bounding box and text
KEYPOINT_COLOR = (0, 255, 0)  # Green for keypoints

try:
    # Face Detector
    model_file = open('./models/blaze_face_short_range.tflite', "rb")
    model_data = model_file.read()
    model_file.close()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.FaceDetectorOptions(base_options=base_options)
    face_detector = vision.FaceDetector.create_from_options(options)

    # Hand Detector
    model_file = open('./models/hand_landmarker.task', "rb")
    model_data = model_file.read()
    model_file.close()
    hand_base_options = python.BaseOptions(model_asset_buffer=model_data)
    hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
except Exception as e:
    print(f"Error initializing FaceDetector: {e}")
    exit()

# Webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create a named window with WINDOW_NORMAL flag to allow resizing
cv2.namedWindow('Face and Hand Detection', cv2.WINDOW_NORMAL)

# Set initial window size to be larger than the default
cv2.resizeWindow('Face and Hand Detection', 800, 600)

# Credit to https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    def is_valid_normalized_value(value: float) -> bool:
        return (value >= 0 or math.isclose(0, value)) and (value <= 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Box and keypoint visualization
def visualize(image: np.ndarray, face_detection_result, hand_detection_result) -> np.ndarray:
    """Draws bounding boxes, keypoints, and labels for faces and hands on the input image.
    Args:
        image: Input BGR image (from OpenCV).
        face_detection_result: MediaPipe FaceDetector detection result object.
        hand_detection_result: MediaPipe HandLandmarker detection result object.
    Returns:
        Annotated image with bounding boxes, keypoints, and labels for faces and hands.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    # Visualize face detections
    for detection in face_detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            if keypoint_px:
                cv2.circle(annotated_image, keypoint_px, 2, KEYPOINT_COLOR, 2)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name or 'Face'
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS
        )

    # Visualize hand detections
    if hand_detection_result and hand_detection_result.handedness:
        for idx, hand_landmarks in enumerate(hand_detection_result.hand_landmarks):
            # Get handedness
            handedness = hand_detection_result.handedness[idx][0]
            category_name = handedness.category_name or 'Unknown'
            probability = round(handedness.score, 2)

            # Calculate bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            start_point = _normalized_to_pixel_coordinates(x_min, y_min, width, height)
            end_point = _normalized_to_pixel_coordinates(x_max, y_max, width, height)

            if start_point and end_point:
                cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

                # Draw label and score
                result_text = f"{category_name} ({probability})"
                text_location = (MARGIN + start_point[0], MARGIN + ROW_SIZE + start_point[1])
                cv2.putText(
                    annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS
                )

            # Draw hand landmarks
            for landmark in hand_landmarks:
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                if landmark_px:
                    cv2.circle(annotated_image, landmark_px, 2, KEYPOINT_COLOR, 2)

    return annotated_image

# Webcam processing loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect faces and hands in the frame
        face_detection_result = face_detector.detect(mp_image)
        hand_detection_result = hand_detector.detect(mp_image)

        # Visualize both face and hand detection results
        annotated_image = visualize(frame, face_detection_result, hand_detection_result)

        # Display the annotated frame
        cv2.imshow('Face and Hand Detection', annotated_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error during processing: {e}")

# Release resources when closed
cap.release()
cv2.destroyAllWindows()