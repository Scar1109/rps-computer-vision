"""
gesture_utils.py — Hand Detection & Gesture Classification

Uses the MediaPipe Tasks Hand Landmarker API to detect hand landmarks
and classify gestures into Rock, Paper, or Scissors using simple
rule-based finger position logic.
"""

import os
import cv2
import urllib.request
import mediapipe as mp
from collections import deque, Counter

# ──────────────────────────────────────────────────────────────
# MediaPipe Tasks API setup
# ──────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model download URL (official Google-hosted model)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)

# ──────────────────────────────────────────────────────────────
# Landmark indices
# ──────────────────────────────────────────────────────────────
# Tip = very end of finger, PIP = middle knuckle, MCP/CMC = base
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [1, 5,  9, 13, 17]

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                   # Palm base
]

# ──────────────────────────────────────────────────────────────
# Gesture stabilization buffer (majority vote over last N frames)
# ──────────────────────────────────────────────────────────────
BUFFER_SIZE = 7
gesture_buffer = deque(maxlen=BUFFER_SIZE)


def ensure_model_downloaded():
    """Download the hand landmarker model (~10 MB) if not present."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading hand landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded!")


# ──────────────────────────────────────────────────────────────
# Finger state detection
# ──────────────────────────────────────────────────────────────
def _get_finger_states(landmarks) -> list[bool]:
    """
    Determine which fingers are extended (True) or folded (False).

    Thumb  → compare x-distance of tip vs IP relative to CMC base.
    Others → tip above PIP (lower y) means extended.

    Returns: [thumb, index, middle, ring, pinky]
    """
    states = []
    for i in range(5):
        tip = landmarks[FINGER_TIPS[i]]
        pip_joint = landmarks[FINGER_PIPS[i]]

        if i == 0:  # Thumb uses x-axis comparison
            mcp = landmarks[FINGER_MCPS[i]]
            is_extended = abs(tip.x - mcp.x) > abs(pip_joint.x - mcp.x)
        else:  # Other fingers: tip.y < pip.y means extended
            is_extended = tip.y < pip_joint.y

        states.append(is_extended)
    return states


def classify_gesture(landmarks) -> str:
    """
    Classify hand gesture from finger states.

    Rock     → 0-1 fingers extended (fist)
    Paper    → 4-5 fingers extended (open hand)
    Scissors → index + middle up, ring + pinky down
    """
    states = _get_finger_states(landmarks)
    extended = sum(states)
    _, index, middle, ring, pinky = states

    if extended >= 4:
        return "Paper"
    if index and middle and not ring and not pinky:
        return "Scissors"
    if extended <= 1:
        return "Rock"
    return "Unknown"


def get_stabilized_gesture(landmarks) -> str:
    """Classify gesture with majority-vote smoothing over recent frames."""
    raw = classify_gesture(landmarks)
    gesture_buffer.append(raw)
    return Counter(gesture_buffer).most_common(1)[0][0]


def clear_gesture_buffer():
    """Clear the stabilization buffer."""
    gesture_buffer.clear()


# ──────────────────────────────────────────────────────────────
# Drawing utilities
# ──────────────────────────────────────────────────────────────
def draw_landmarks_on_frame(frame, landmarks, color=(102, 234, 126)):
    """Draw hand skeleton (lines + dots) on the frame."""
    h, w, _ = frame.shape

    # Lines between connected landmarks
    for s, e in HAND_CONNECTIONS:
        pt1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
        pt2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    # Dots at each landmark
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 6, color, 2)

    return frame


# ──────────────────────────────────────────────────────────────
# Hand detector factory
# ──────────────────────────────────────────────────────────────
def create_hand_detector():
    """
    Create a MediaPipe HandLandmarker in VIDEO mode.
    Downloads the model automatically on first use.
    """
    ensure_model_downloaded()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)
