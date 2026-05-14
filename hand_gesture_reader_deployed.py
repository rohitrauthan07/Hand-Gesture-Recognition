# Author: Odilbek Tokhirov
# run the following command in terminal:
# streamlit run hand_gesture_reader_deployed.py

import os
import urllib.request
import warnings
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui as pag
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# initialization
model_name_rf = 'model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl'
model_name_xgb = 'model_XGBoost__date_time_2023_09_23__11_55_17__acc_1.0__hand__oneimage.pkl'
model_name_nn = ''
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)


def _patch_legacy_sklearn_trees(estimator):
    """Joblib dumps from sklearn<1.4 lack monotonic_cst; sklearn>=1.4 expects it on trees."""
    trees = getattr(estimator, 'estimators_', None)
    if trees is None:
        return
    for tree in np.ravel(trees):
        if tree is not None and not hasattr(tree, 'monotonic_cst'):
            tree.monotonic_cst = None


model = joblib.load(model_name_rf)
_patch_legacy_sklearn_trees(model)
idx_to_class = {
    0: 'Closed',
    1: 'Three',
    2: 'Open',
    3: 'Zero',
}
class_to_key = {
    'Closed': 'up',
    'Three': 'right',
    'Open': 'left',
    'Zero': 'down',
}
n_classes = len(class_to_key)

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


hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(_ensure_hand_landmarker_model())),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmark_spec = vision.drawing_utils.DrawingSpec(
    color=(255, 0, 255), thickness=4, circle_radius=2
)
connection_spec = vision.drawing_utils.DrawingSpec(
    color=(20, 180, 90), thickness=2, circle_radius=2
)

# streamlit code
st.title('Hand Gesture Reader')
frame_placeholder = st.empty()
st.subheader('Predicted Probabilities:')
text_boxes = [col.empty() for col in st.columns(n_classes)]

st.subheader('Activated Keys:')
key_inputs = []
for key, col in zip(class_to_key, st.columns(n_classes)):
    with col:
        key_input = st.text_input(f'Hand {key}', class_to_key[key])
        key_inputs.append(key_input)

# description section
description = '''
This is a hand gesture recognition app. It uses the webcam to recognize hand gestures of the right hand and performs actions corresponding
to the gestures. When your hand is shown to the camera, the recognized gesture will be displayed and a certain key will
be activated. You can change the type of key that is activated. Just type the key name above in the gesture category to which
you wish to assign. Please not that the page will refresh after updating a key. You can refer to pyautogui documentation on how to write certain keys (for e.g. F1 key).
The sample gestures are given below:
'''
st.header('Description')
st.markdown(description)
for i in range(0, n_classes, 2):
    for col, j in zip(st.columns(2), range(2)):
        with col:
            image_name = os.listdir('sample_images')[i + j]
            st.image(os.path.join('sample_images', image_name), caption=image_name[:-4], width=320)

current_command = None
capture = cv2.VideoCapture(0)  # 0 integrated | 1 plugged
frame_timestamp_ms = 0

with vision.HandLandmarker.create_from_options(hand_options) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        height, width = frame.shape[:-1]
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected = hands.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        if detected.hand_landmarks:
            for hand_lm in detected.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    image,
                    hand_lm,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )

            first_hand = detected.hand_landmarks[0]
            x = []
            for lm in first_hand:
                x.extend([lm.x, lm.y])

            x = np.array(x)
            x_max = int(width * np.max(x[::2]))
            x_min = int(width * np.min(x[::2]))
            y_max = int(height * np.max(x[1::2]))
            y_min = int(height * np.min(x[1::2]))
            x = x[None, :]
            predicted_class_probabilities = model.predict_proba(x)[0]

            for text_box, probability, i in zip(
                text_boxes, predicted_class_probabilities, range(n_classes)
            ):
                text_box.text(f'Hand {list(class_to_key.keys())[i]} = {probability}')

            if np.max(predicted_class_probabilities) > 0.5:
                yhat_idx = int(np.argmax(predicted_class_probabilities))
                yhat = idx_to_class[yhat_idx]
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(
                    image,
                    f'{yhat}',
                    (50, 50),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if current_command != yhat:
                    pag.press(key_inputs[yhat_idx])
                    current_command = yhat

        elif current_command is not None:
            current_command = None

        frame_placeholder.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels='RGB'
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
