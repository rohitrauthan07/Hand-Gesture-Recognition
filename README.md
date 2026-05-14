# Hand Gesture Recognition

Real-time hand tracking and gesture-driven control using **MediaPipe Hand Landmarker** (Tasks API), **OpenCV**, and optional **scikit-learn** classifiers. The repo includes a finger-counting demo, a desktop gesture-to-keyboard bridge, and a **Streamlit** variant with configurable hotkeys.

---

## What this project does

| Component | Purpose |
|-----------|---------|
| **`hand_tracker.py`** | Webcam loop: detects up to two hands, draws landmarks, counts extended fingers per hand, shows a **total finger count**, and can capture frame sequences to `data/<subdir>/sequenceN/` when you press **`r`** (quit with **`q`**). The MediaPipe `.task` model is downloaded automatically on first run if missing. |
| **`hand_gesture_reader.py`** | Uses landmarks from the first detected hand as a 42-dimensional feature vector, runs a **Random Forest** classifier (`model_rf__...pkl`), maps classes **Closed / Three / Open / Zero** to **arrow keys** via **PyAutoGUI**, and only sends a key when the predicted class changes. |
| **`hand_gesture_reader_deployed.py`** | Same core logic wrapped in **Streamlit**: live video in the app, per-class **probability** readouts, **user-editable key bindings**, and an in-app description (expects a `sample_images/` folder for gesture thumbnails). |

**Gesture → default key mapping**

| Gesture | Default key |
|---------|-------------|
| Closed | Up |
| Three | Right |
| Open | Left |
| Zero | Down |

**Author credit** (gesture reader scripts): Odilbek Tokhirov.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Gesture reader scripts** expect trained model files in the project root (for example `model_rf__date_time_2023_09_23__12_22_48__acc_1.0__hand__oneimage.pkl`). If those files are not present, add them or train your own classifier compatible with the 42-feature layout. The Streamlit app also references a `sample_images/` directory for the description gallery.

---

## How to run

**Finger counting + optional dataset capture**

```bash
python hand_tracker.py
```

**Desktop gesture reader (OpenCV window)**

```bash
python hand_gesture_reader.py
```

**Streamlit deployment**

```bash
streamlit run hand_gesture_reader_deployed.py
```

The **Hand Landmarker** model is stored under `models/hand_landmarker.task` and is fetched from Google Cloud Storage the first time it is needed.

---

## Repository layout

- `hand_tracker.py` — landmark visualization, finger counting, sequence capture hotkey **`r`**
- `hand_gesture_reader.py` — RF prediction + PyAutoGUI key presses
- `hand_gesture_reader_deployed.py` — Streamlit UI + configurable keys + `predict_proba` thresholding
- `models/` — `hand_landmarker.task` (MediaPipe)
- `assets/` — reference images for counts (see below)
- `requirements.txt` — Python dependencies

---

## Assets (finger count examples)

These images illustrate counting results shown in the tracker-style workflow (0, 1, 4, 5, and 10 fingers total across hands).

| Count: 0 | Count: 1 |
|:--------:|:--------:|
| ![Count 0](assets/count%200.png) | ![Count 1](assets/count%201.png) |

| Count: 4 | Count: 5 |
|:--------:|:--------:|
| ![Count 4](assets/count%204.png) | ![Count 5](assets/count%205.png) |

| Count: 10 |
|:---------:|
| ![Count 10](assets/count%2010.png) |

---

## Technical notes

- **MediaPipe Tasks** `HandLandmarker` runs in **VIDEO** mode with timestamps for temporal stability.
- **`hand_tracker.py`** implements finger counting from landmark geometry (PIP vs tip for four fingers; thumb uses handedness-aware horizontal checks).
- **sklearn ≥ 1.4**: legacy joblib trees are patched at load time (`monotonic_cst`) so older RF dumps keep working.
- **NumPy** is pinned to `<2` in `requirements.txt` for compatibility with the scientific stack used by the classifiers.

---

## License

Add a `LICENSE` file if you plan to distribute this repository; none is included in the current tree.
