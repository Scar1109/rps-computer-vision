import cv2
import time
import numpy as np
import mediapipe as mp

from gesture_utils import (
    create_hand_detector,
    draw_landmarks_on_frame,
    get_stabilized_gesture,
    clear_gesture_buffer,
)
from game_logic import (
    get_ai_move,
    determine_winner,
    MOVE_EMOJIS,
)

# ──────────────────────────────────────────────────────────────
# Color palette (BGR format for OpenCV)
# ──────────────────────────────────────────────────────────────
BG_DARK       = (30, 30, 40)
TEXT_WHITE     = (240, 240, 240)
TEXT_GRAY      = (160, 160, 160)
ACCENT_PURPLE  = (234, 126, 102)   # Soft purple
ACCENT_GREEN   = (129, 185, 16)    # Green
ACCENT_RED     = (68, 68, 239)     # Red
ACCENT_YELLOW  = (11, 158, 245)    # Amber/yellow
PANEL_COLOR    = (40, 40, 55)

# ──────────────────────────────────────────────────────────────
# Helper: draw a semi-transparent filled rectangle
# ──────────────────────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, color=PANEL_COLOR, alpha=0.7):
    """Draw a semi-transparent rectangle on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text_centered(frame, text, y, font_scale=1.0, color=TEXT_WHITE, thickness=2):
    """Draw text horizontally centered on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────
def main():
    # ── Initialize ───────────────────────────────────────
    print("🎮 Rock Paper Scissors AI")
    print("   SPACE = Play Round | R = Reset | Q/ESC = Quit\n")

    detector = create_hand_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Set camera resolution for quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Game state
    scores = {"player": 0, "ai": 0, "draws": 0, "rounds": 0}
    last_result = None          # {"player": ..., "ai": ..., "result": ...}
    history = []                # List of recent round dicts
    countdown_active = False
    countdown_start = 0
    gesture_at_capture = None   # Gesture locked at countdown end
    frame_index = 0             # Monotonic frame counter for MediaPipe
    result_flash_time = 0       # Timestamp when result was shown

    window_name = "Rock Paper Scissors AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── MediaPipe hand detection ─────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_index += 1
        timestamp_ms = frame_index * 33  # ~30 fps

        try:
            result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            result = None

        current_gesture = "No Hand"
        hand_detected = False

        if result and result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                draw_landmarks_on_frame(frame, hand_lm)
                current_gesture = get_stabilized_gesture(hand_lm)
                hand_detected = True
                break

        # ── Top bar: title + detected gesture ────────────
        draw_panel(frame, 0, 0, w, 70, BG_DARK, 0.75)
        cv2.putText(frame, "Rock Paper Scissors AI", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, ACCENT_PURPLE, 3, cv2.LINE_AA)

        # Current gesture badge (top right)
        gesture_label = "Detected:" if hand_detected else "Status:"
        gesture_value = current_gesture
        gesture_color = ACCENT_GREEN if hand_detected else TEXT_GRAY
        badge_text = f"{gesture_label} {gesture_value}"
        (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(frame, badge_text, (w - tw - 25, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, gesture_color, 2, cv2.LINE_AA)

        # Status dot (green = hand found, red = no hand)
        dot_color = (0, 200, 100) if hand_detected else (0, 0, 200)
        cv2.circle(frame, (w - tw - 42, 42), 7, dot_color, -1)

        # ── Scoreboard panel (top-left below title bar) ──
        draw_panel(frame, 10, 82, 280, 145, BG_DARK, 0.72)
        cv2.putText(frame, "SCOREBOARD", (25, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_PURPLE, 2, cv2.LINE_AA)
        # Divider line under title
        cv2.line(frame, (25, 118), (270, 118), (80, 80, 100), 1, cv2.LINE_AA)

        cv2.putText(frame, "Player", (25, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, str(scores['player']), (230, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_GREEN, 2, cv2.LINE_AA)

        cv2.putText(frame, "AI", (25, 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, str(scores['ai']), (230, 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_RED, 2, cv2.LINE_AA)

        cv2.putText(frame, "Draws", (25, 208),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, str(scores['draws']), (230, 208),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_YELLOW, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Round {scores['rounds']}", (140, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_GRAY, 1, cv2.LINE_AA)

        # ── Countdown overlay ────────────────────────────
        if countdown_active:
            elapsed = time.time() - countdown_start

            if elapsed < 3.0:
                # Phase 1: Count up 1 → 2 → 3 (one number per second)
                count_num = int(elapsed) + 1  # 1, 2, 3
                draw_panel(frame, w // 2 - 80, h // 2 - 80, 160, 160, BG_DARK, 0.8)
                put_text_centered(frame, str(count_num), h // 2 + 30,
                                  font_scale=4.0, color=ACCENT_YELLOW, thickness=6)
                put_text_centered(frame, "Get ready!", h // 2 + 70,
                                  font_scale=0.8, color=TEXT_GRAY, thickness=1)

            elif elapsed < 4.0:
                # Phase 2: "SHOW!" — hold your gesture, it's being captured
                draw_panel(frame, w // 2 - 120, h // 2 - 80, 240, 160, BG_DARK, 0.8)
                put_text_centered(frame, "SHOW!", h // 2 + 30,
                                  font_scale=3.0, color=ACCENT_GREEN, thickness=6)
                put_text_centered(frame, "Hold your gesture!", h // 2 + 70,
                                  font_scale=0.7, color=TEXT_WHITE, thickness=1)

            else:
                # Phase 3: Capture the gesture and play the round
                countdown_active = False
                gesture_at_capture = current_gesture

                if gesture_at_capture in ("Rock", "Paper", "Scissors"):
                    ai_move = get_ai_move()
                    result_str = determine_winner(gesture_at_capture, ai_move)

                    scores["rounds"] += 1
                    if result_str == "Win":
                        scores["player"] += 1
                    elif result_str == "Lose":
                        scores["ai"] += 1
                    else:
                        scores["draws"] += 1

                    last_result = {
                        "player": gesture_at_capture,
                        "ai": ai_move,
                        "result": result_str,
                    }
                    result_flash_time = time.time()

                    # Add to history (keep last 5)
                    history.append(last_result.copy())
                    if len(history) > 5:
                        history.pop(0)
                else:
                    last_result = {"player": "???", "ai": "—", "result": "No hand!"}
                    result_flash_time = time.time()

        # ── Last result panel (right side) ───────────────
        if last_result:
            seconds_since = time.time() - result_flash_time
            panel_alpha = max(0.55, 0.85 - seconds_since * 0.04)

            panel_w = 340
            panel_h = 220
            px = w - panel_w - 15
            py = 82

            # Pick panel accent color based on result
            if last_result["result"] == "Win":
                accent = ACCENT_GREEN
            elif last_result["result"] == "Lose":
                accent = ACCENT_RED
            else:
                accent = ACCENT_YELLOW

            draw_panel(frame, px, py, panel_w, panel_h, BG_DARK, panel_alpha)

            # Result headline
            result_labels = {"Win": "YOU WIN!", "Lose": "AI WINS!", "Draw": "IT'S A DRAW!", "No hand!": "NO HAND!"}
            result_text = result_labels.get(last_result["result"], last_result["result"])
            cv2.putText(frame, result_text, (px + 20, py + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, accent, 3, cv2.LINE_AA)

            # Divider
            cv2.line(frame, (px + 20, py + 58), (px + panel_w - 20, py + 58),
                     (80, 80, 100), 1, cv2.LINE_AA)

            # Player move
            cv2.putText(frame, "Your move:", (px + 20, py + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_GRAY, 1, cv2.LINE_AA)
            cv2.putText(frame, last_result['player'], (px + 170, py + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_WHITE, 2, cv2.LINE_AA)

            # AI move
            cv2.putText(frame, "AI move:", (px + 20, py + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_GRAY, 1, cv2.LINE_AA)
            cv2.putText(frame, last_result['ai'], (px + 170, py + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_WHITE, 2, cv2.LINE_AA)

            # Round info
            cv2.putText(frame, f"Round {scores['rounds']}", (px + 20, py + 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_GRAY, 1, cv2.LINE_AA)

        # ── Recent history (bottom-left) ─────────────────
        if history:
            row_h = 32
            hist_y = h - row_h * len(history) - 65
            draw_panel(frame, 10, hist_y - 30, 380, row_h * len(history) + 45, BG_DARK, 0.65)
            cv2.putText(frame, "RECENT ROUNDS", (25, hist_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT_PURPLE, 1, cv2.LINE_AA)

            for i, entry in enumerate(reversed(history)):
                ey = hist_y + 20 + i * row_h
                rnd = scores["rounds"] - i
                color = (ACCENT_GREEN if entry["result"] == "Win"
                         else ACCENT_RED if entry["result"] == "Lose"
                         else ACCENT_YELLOW)
                line = f"R{rnd}:  {entry['player']:>8}  vs  {entry['ai']:<8}  {entry['result']}"
                cv2.putText(frame, line, (25, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ── Center prompt when idle ──────────────────────
        if not countdown_active and not last_result:
            put_text_centered(frame, "Press SPACE to play!", h // 2 + 10,
                              font_scale=1.2, color=TEXT_GRAY, thickness=2)

        # ── Bottom bar: controls hint ────────────────────
        draw_panel(frame, 0, h - 45, w, 45, BG_DARK, 0.72)
        cv2.putText(frame, "[SPACE] Play Round    [R] Reset Scores    [Q] Quit",
                    (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_GRAY, 1, cv2.LINE_AA)

        # ── Show the frame ───────────────────────────────
        cv2.imshow(window_name, frame)

        # ── Keyboard input ───────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord(" ") and not countdown_active:  # SPACE
            countdown_active = True
            countdown_start = time.time()
        elif key == ord("r"):  # R = reset
            scores = {"player": 0, "ai": 0, "draws": 0, "rounds": 0}
            last_result = None
            history.clear()
            clear_gesture_buffer()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n👋 Thanks for playing!")


if __name__ == "__main__":
    main()
