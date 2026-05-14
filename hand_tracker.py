import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

subdir = 'hand_down'                
n_frames_save = 20
iteration_counter = n_frames_save + 1
folder_counter = 1
save_seq_index = None

_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)


def _ensure_hand_landmarker_model() -> Path:
    model_path = Path(__file__).resolve().parent / 'models' / 'hand_landmarker.task'
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading hand_landmarker.task (one-time, ~few MB)...')
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


def count_extended_fingers(landmarks, is_right_hand: bool) -> int:
    """Count raised fingers from 21 hand landmarks (MediaPipe order)."""
    lm = landmarks
    n = 0
    # Index, middle, ring, pinky: tip above PIP in image coords (y grows downward)
    if lm[8].y < lm[6].y:
        n += 1
    if lm[12].y < lm[10].y:
        n += 1
    if lm[16].y < lm[14].y:
        n += 1
    if lm[20].y < lm[18].y:
        n += 1
    # Thumb: compare tip (4) to IP (3); side depends on handedness
    if is_right_hand:
        if lm[4].x < lm[3].x:
            n += 1
    else:
        if lm[4].x > lm[3].x:
            n += 1
    return n


def _hand_is_right(handedness_per_hand) -> bool:
    if not handedness_per_hand:
        return True
    cat = handedness_per_hand[0]
    name = (cat.category_name or cat.display_name or '').lower()
    return 'right' in name


model_path = _ensure_hand_landmarker_model()
hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmark_spec = vision.drawing_utils.DrawingSpec(
    color=(255, 0, 255), thickness=4, circle_radius=2
)
connection_spec = vision.drawing_utils.DrawingSpec(
    color=(20, 180, 90), thickness=2, circle_radius=2
)

capture = cv2.VideoCapture(0)
frame_timestamp_ms = 0

with vision.HandLandmarker.create_from_options(hand_options) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected = hands.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        h, w, _ = image.shape
        total_fingers = 0
        if detected.hand_landmarks:
            for idx, hand_lm in enumerate(detected.hand_landmarks):
                vision.drawing_utils.draw_landmarks(
                    image,
                    hand_lm,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )
                hs = detected.handedness[idx] if idx < len(detected.handedness) else []
                is_right = _hand_is_right(hs)
                n_fingers = count_extended_fingers(hand_lm, is_right)
                total_fingers += n_fingers
                wrist = hand_lm[0]
                wx = int(wrist.x * w)
                wy = int(wrist.y * h)
                cv2.putText(
                    image,
                    str(n_fingers),
                    (wx - 10, wy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        cv2.putText(
            image,
            f'Count: {total_fingers}',
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.imshow('Webcam', image)

        if cv2.waitKey(10) & 0xFF == ord('r'):
            save_seq_index = folder_counter
            seq_folder_path = os.path.join('data', subdir, f'sequence{folder_counter}')
            os.makedirs(os.path.join('data', subdir), exist_ok=True)
            os.mkdir(seq_folder_path)
            folder_counter += 1
            iteration_counter = 1

        if iteration_counter < n_frames_save + 1 and save_seq_index is not None:
            seq_folder_path = os.path.join('data', subdir, f'sequence{save_seq_index}')
            cv2.imwrite(
                os.path.join(
                    seq_folder_path,
                    f'{subdir}_sequence{save_seq_index}_frame{iteration_counter}.jpg',
                ),
                image,
            )
            if iteration_counter == n_frames_save:
                print(f'Images for sequence {save_seq_index} saved.')
            iteration_counter += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
