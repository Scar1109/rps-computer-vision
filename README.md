# ✊📄✂️ Rock Paper Scissors AI

> **A real-time computer vision game that detects your hand gestures through a webcam and plays Rock-Paper-Scissors against you — no model training required.**

Built with **Python**, **OpenCV**, and **Google MediaPipe** as a portfolio project showcasing practical computer vision, real-time image processing, and clean software architecture.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-FF6F00?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What This Project Demonstrates

| Skill Area | What's Shown |
|---|---|
| **Computer Vision** | Real-time hand landmark detection using MediaPipe's 21-point hand model |
| **Image Processing** | Frame manipulation, coordinate normalization, and overlay rendering with OpenCV |
| **Algorithm Design** | Rule-based gesture classification using geometric finger-state analysis |
| **Signal Processing** | Majority-vote stabilization buffer for noise reduction across frames |
| **Software Architecture** | Clean separation of concerns: detection → classification → game logic → UI |
| **Python Engineering** | Modular code, docstrings, type hints, and beginner-friendly structure |

---

## ✨ Features

- 🎥 **Full-quality webcam feed** with real-time hand skeleton overlay
- 🤖 **AI opponent** with instant random move generation
- ⏱️ **1-2-3-SHOW! countdown** before each round capture
- 📊 **Live scoreboard** — player wins, AI wins, draws, total rounds
- 📜 **Round history** — last 5 rounds displayed on-screen
- 🎯 **Gesture stabilization** — majority-vote buffer prevents flickering
- ⌨️ **Keyboard controls** — Space to play, R to reset, Q to quit
- 🧠 **No ML training** — works out of the box with rule-based logic

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.9+** | Core language |
| **OpenCV** | Webcam capture, frame processing, on-screen UI rendering |
| **MediaPipe** | Hand landmark detection (21 keypoints per hand) |
| **NumPy** | Array operations for image processing |

---

## 📁 Project Structure

```
Rock-Paper-Scissors-AI/
├── app.py               # Main game loop — camera, overlays, keyboard controls
├── gesture_utils.py     # Hand detection, finger-state analysis, gesture classification
├── game_logic.py        # AI move generation, winner determination, round formatting
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** and **pip**
- A webcam (built-in or USB)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Rock-Paper-Scissors-AI.git
cd Rock-Paper-Scissors-AI

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

> 📥 On first launch, the MediaPipe hand model (~10 MB) downloads automatically.

---

## 🎮 Controls

| Key | Action |
|---|---|
| **SPACE** | Start 1-2-3-SHOW! countdown → captures your gesture |
| **R** | Reset all scores to zero |
| **Q** / **ESC** | Quit the application |

### How to Play
1. Run `python app.py` — a camera window opens
2. Hold your hand in front of the camera
3. Press **SPACE** — a **1, 2, 3, SHOW!** countdown begins
4. Show ✊ Rock, 📄 Paper, or ✂️ Scissors and **hold** during "SHOW!"
5. The AI reveals its move and the winner is displayed

---

## 🧠 How Gesture Detection Works

The system uses **MediaPipe Hands** to track 21 hand landmarks per frame, then applies geometric rules to classify the gesture.

### Step 1: Finger State Analysis
For each of the 5 fingers, the system compares the **fingertip** position to the **PIP joint** (middle knuckle):

- **Extended** → Tip is above the PIP joint (lower y-coordinate)
- **Folded** → Tip is below the PIP joint
- **Thumb** → Uses x-axis distance comparison (relative to CMC joint)

### Step 2: Gesture Classification Rules

| Gesture | Condition |
|---|---|
| ✊ **Rock** | 0–1 fingers extended |
| 📄 **Paper** | 4–5 fingers extended |
| ✂️ **Scissors** | Index + Middle extended, Ring + Pinky folded |

### Step 3: Stabilization
A **majority-vote buffer** (last 7 frames) smooths out single-frame errors, providing stable, reliable predictions during the countdown.

---

## 🔮 Future Improvements

- [ ] Sound effects on win/lose/draw
- [ ] Animated visual effects on result reveal
- [ ] Best-of-N match mode with auto-progression
- [ ] Win-rate statistics and session charts
- [ ] Two-player split-screen mode
- [ ] Extended gestures (Lizard, Spock)
- [ ] Configurable countdown speed

---

## 📄 License

Open source under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Python, OpenCV & MediaPipe<br>
  <sub>A portfolio project by <a href="https://github.com/yourusername">Your Name</a></sub>
</p>
