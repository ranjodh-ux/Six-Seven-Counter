"""
SIX-SEVEN COUNTER
-----------------
Pick a timer (5 / 10 / 20 / 30 sec), get a 3-2-1 countdown, then move your
hands up and down in opposite directions. Every time one hand goes high while
the other goes low, you get +1.

Install:  pip install opencv-python mediapipe numpy
Run:      python six_seven_counter.py

Controls:
  Menu     -> click a button or press 1 / 2 / 3 / 4   (Q or ESC = quit)
  In round -> ESC = abort
  Results  -> R = retry, M = menu, Q / ESC = quit
"""

import json
import os
import sys
import time
import urllib.request

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"Missing package: {e}")
    print("Run: pip install opencv-python mediapipe numpy")
    sys.exit(1)

# ----------------------------- config -----------------------------------
TIMERS = [5, 10, 20, 30]
WIN = "six-seven counter"
RECORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "six_seven_records.json")

# How far apart (as a fraction of frame height) the two hands must be
# vertically to count as "one up, one down". Raise it if it counts too easily.
HEIGHT_GAP = 0.12

# ONE tracked dot per hand: middle-finger knuckle (palm center)
DOT_IDS = (9,)

GREEN = (80, 230, 100)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
RED = (60, 60, 255)


# ----------------------------- records ----------------------------------
def load_records():
    records = {str(t): 0 for t in TIMERS}
    try:
        if os.path.exists(RECORD_FILE):
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for t in TIMERS:
                records[str(t)] = int(data.get(str(t), 0))
    except (OSError, ValueError, TypeError, AttributeError) as e:
        print(f"[warn] Couldn't read records ({e}). Starting fresh.")
    return records


def save_records(records):
    try:
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except OSError as e:
        print(f"[warn] Couldn't save records: {e}")


# ----------------------------- drawing ----------------------------------
def put(img, text, org, scale=1.0, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_center(img, text, cy, scale=1.0, color=WHITE, thick=2, cx=None):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx = img.shape[1] // 2 if cx is None else cx
    put(img, text, (cx - w // 2, cy + h // 2), scale, color, thick)


def draw_hands(frame, hands_info):
    for h in hands_info:
        for (x, y) in h["pts"]:
            cv2.circle(frame, (x, y), 16, (0, 0, 0), -1)
            cv2.circle(frame, (x, y), 12, GREEN, -1)


def window_closed():
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# ----------------------------- camera / tracking ------------------------
def open_camera():
    for idx in (0, 1, 2):
        for backend in (None, getattr(cv2, "CAP_DSHOW", None)):
            try:
                cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ok, _ = cap.read()
                    if ok:
                        return cap
                cap.release()
            except Exception:
                continue
    raise RuntimeError("No working camera found. Check it's connected and not used by another app.")


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    """Download the hand model once (needs internet the first time only)."""
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 1_000_000:
        return MODEL_FILE
    print("Downloading hand model (one time, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as e:
        try:
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
        except OSError:
            pass
        raise RuntimeError(
            f"Couldn't download the hand model ({e}).\n"
            f"Download it manually from:\n{MODEL_URL}\n"
            f"and put it next to this script as 'hand_landmarker.task'."
        )
    return MODEL_FILE


def make_landmarker(model_path):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def grab(cap, hands):
    """Read a mirrored frame and return (frame, [hand dicts])."""
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Lost connection to the camera.")
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = hands.detect_for_video(mp_img, int(time.monotonic() * 1000))
    info = []
    for lms in res.hand_landmarks or []:
        pts = [(int(lms[i].x * w), int(lms[i].y * h)) for i in DOT_IDS]
        ny = sum(lms[i].y for i in DOT_IDS) / len(DOT_IDS)   # 0 = top
        nx = sum(lms[i].x for i in DOT_IDS) / len(DOT_IDS)
        info.append({"x": nx, "y": ny, "pts": pts})
    return frame, info


# ----------------------------- screens ----------------------------------
def menu_screen(records):
    """Returns chosen seconds, or None to quit."""
    W, H = 960, 540
    btn_w, btn_h, gap = 190, 120, 25
    total = len(TIMERS) * btn_w + (len(TIMERS) - 1) * gap
    x0, y0 = (W - total) // 2, 250
    rects = [(x0 + i * (btn_w + gap), y0, btn_w, btn_h) for i in range(len(TIMERS))]
    click = {"pos": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pos"] = (x, y)

    cv2.setMouseCallback(WIN, on_mouse)
    try:
        while True:
            img = np.full((H, W, 3), 25, np.uint8)
            put_center(img, "SIX-SEVEN COUNTER", 80, 1.8, YELLOW, 4)
            put_center(img, "Pick a timer", 150, 1.0, WHITE, 2)
            for i, (t, (x, y, w, h)) in enumerate(zip(TIMERS, rects)):
                cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 3)
                put_center(img, f"{t} sec", y + 45, 1.3, WHITE, 3, cx=x + w // 2)
                put_center(img, f"Best: {records[str(t)]}", y + 90, 0.8, YELLOW, 2, cx=x + w // 2)
                put_center(img, f"[{i + 1}]", y + h + 30, 0.7, (180, 180, 180), 1, cx=x + w // 2)
            put_center(img, "Q / ESC to quit", H - 30, 0.6, (160, 160, 160), 1)
            cv2.imshow(WIN, img)

            key = cv2.waitKey(30) & 0xFF
            if window_closed() or key in (27, ord("q")):
                return None
            if ord("1") <= key <= ord("0") + len(TIMERS):
                return TIMERS[key - ord("1")]
            if click["pos"]:
                cx, cy = click["pos"]
                click["pos"] = None
                for t, (x, y, w, h) in zip(TIMERS, rects):
                    if x <= cx <= x + w and y <= cy <= y + h:
                        return t
    finally:
        cv2.setMouseCallback(WIN, lambda *a: None)


def run_round(cap, hands, seconds, record):
    """Countdown + timed round. Returns (score, last_frame) or (None, None) if aborted."""
    frame = None

    # ---- 3-2-1 countdown (camera already live) ----
    for n in (3, 2, 1):
        end = time.time() + 1.0
        while time.time() < end:
            frame, info = grab(cap, hands)
            draw_hands(frame, info)
            put_center(frame, str(n), frame.shape[0] // 2, 8, YELLOW, 14)
            put_center(frame, f"{seconds}s round - get ready!", 60, 1.1, WHITE, 2)
            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) == 27 or window_closed():
                return None, None

    # ---- the actual round ----
    score, last_pose = 0, None
    start = time.time()
    while True:
        remaining = seconds - (time.time() - start)
        if remaining <= 0:
            break

        frame, info = grab(cap, hands)
        H, W = frame.shape[:2]
        status = "Show BOTH hands"

        if len(info) >= 2:
            # two hands -> left/right on screen by x position
            a, b = sorted(info[:2], key=lambda h: h["x"])
            diff = a["y"] - b["y"]          # negative = left hand higher
            pose = None
            if diff < -HEIGHT_GAP:
                pose, status = "L", "Left UP / Right DOWN"
            elif diff > HEIGHT_GAP:
                pose, status = "R", "Right UP / Left DOWN"
            else:
                status = "Keep going..."
            # count only when the pose flips to the opposite one
            if pose and pose != last_pose:
                score += 1
                last_pose = pose

        draw_hands(frame, info)

        # HUD
        put_center(frame, str(score), H // 2 - 20, 7, GREEN, 12)
        put_center(frame, status, H - 40, 0.9, WHITE, 2)
        put(frame, f"Time: {remaining:4.1f}s", (20, 50), 1.1, RED if remaining < 3 else WHITE, 3)
        best_now = max(record, score)
        txt = f"Current record ({seconds}s): {best_now}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        put(frame, txt, (W - tw - 20, 45), 0.9, YELLOW, 2)
        if score > record:
            put(frame, "NEW RECORD!", (W - 260, 85), 0.9, GREEN, 2)

        cv2.imshow(WIN, frame)
        if (cv2.waitKey(1) & 0xFF) == 27 or window_closed():
            return None, None

    return score, frame


def result_screen(last_frame, score, seconds, new_record, record):
    """Returns 'retry', 'menu', or 'quit'."""
    base = last_frame if last_frame is not None else np.full((540, 960, 3), 25, np.uint8)
    img = (base * 0.35).astype(np.uint8)
    H = img.shape[0]
    put_center(img, "TIME'S UP!", int(H * 0.18), 2, YELLOW, 5)
    put_center(img, f"Score: {score}", int(H * 0.38), 3, GREEN, 7)
    if new_record:
        put_center(img, "NEW RECORD!", int(H * 0.55), 1.6, YELLOW, 4)
    else:
        put_center(img, f"Record ({seconds}s): {record}", int(H * 0.55), 1.2, WHITE, 3)
    put_center(img, "[R] Retry    [M] Menu    [Q] Quit", int(H * 0.8), 1.0, WHITE, 2)

    while True:
        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF
        if window_closed() or key in (27, ord("q")):
            return "quit"
        if key == ord("r"):
            return "retry"
        if key == ord("m"):
            return "menu"


# ----------------------------- main -------------------------------------
def main():
    records = load_records()
    cap = None
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)

        hands_ctx = make_landmarker(ensure_model())

        with hands_ctx as hands:
            seconds = None
            while True:
                if seconds is None:
                    seconds = menu_screen(records)
                    if seconds is None:
                        break

                if cap is None:
                    cap = open_camera()

                record = records[str(seconds)]
                score, last_frame = run_round(cap, hands, seconds, record)
                if score is None:        # aborted
                    seconds = None
                    if window_closed():
                        break
                    continue

                new_record = score > record
                if new_record:
                    records[str(seconds)] = score
                    save_records(records)

                choice = result_screen(last_frame, score, seconds, new_record, records[str(seconds)])
                if choice == "quit":
                    break
                if choice == "menu":
                    seconds = None
                # 'retry' keeps the same timer

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[error] {e}")
    except cv2.error as e:
        print(f"[opencv error] {e}")
    except Exception as e:
        print(f"[unexpected error] {type(e).__name__}: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()