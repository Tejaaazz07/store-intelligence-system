"""
track_and_identify.py
=====================
Main video pipeline: YOLO detection → ByteTrack → Face Recognition → Role/Category.

Categories:
  • Lakshana / Mehaa   → ★ Hacker  (Staff / Tech)
  • Shirley            → ★ Hustler (Staff / Sales)
  • Other known staff  → ★ Worker
  • Unregistered face  → Customer (Shopper / Visitor)
"""

import os
import sys
import time
import sqlite3
import argparse
import subprocess
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_recognizer import FaceIdentifier
from person_classifier import (
    PersonClassifier, PersonInfo,
    ROLE_STYLES,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# Minimum confidence (%) to accept a face match as a real identification
MIN_CONFIDENCE_PCT = 85.0

# Motion tracker: how many frames to observe & minimum cumulative bbox movement
MOTION_FRAMES        = 12   # observe over 12 frames
MIN_MOTION_PX        = 8.0  # a real moving person's box centre moves ≥ 8 px total
MOTION_WARMUP_FRAMES = MOTION_FRAMES

# Background subtractor for coarse "is something actually moving here?" check
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120, varThreshold=40, detectShadows=False
)

# ── Voice ──────────────────────────────────────────────────────────────────────
last_speech_time = {}

def announce_voice(title, name):
    now = time.time()
    key = f"{title}_{name}"
    if key not in last_speech_time or (now - last_speech_time[key]) > 8.0:
        last_speech_time[key] = now
        try:
            cmd = (f'Add-Type -AssemblyName System.Speech; '
                   f'(New-Object System.Speech.Synthesis.SpeechSynthesizer)'
                   f'.Speak("{title} {name} identified")')
            subprocess.Popen(['powershell', '-Command', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

# ── Database ───────────────────────────────────────────────────────────────────
def log_event_to_db(event_type, person_id, zone="store", role="customer"):
    try:
        conn = sqlite3.connect("store.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT, person_id TEXT,
                zone TEXT, role TEXT DEFAULT 'customer',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "INSERT INTO events (event_type, person_id, zone, role, timestamp) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (event_type, person_id, zone, role)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

# ── Motion helper ──────────────────────────────────────────────────────────────
def _box_centre(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def _total_path(positions):
    """Sum of Euclidean distances between consecutive positions."""
    pts = list(positions)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(np.array(pts, dtype=float), axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

def _motion_in_mask(fg_mask, x1, y1, x2, y2):
    """Return fraction of foreground (moving) pixels inside a bounding box."""
    region = fg_mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return np.count_nonzero(region) / region.size

# ── Drawing helpers ────────────────────────────────────────────────────

def _draw_person(frame, x1, y1, x2, y2, person: PersonInfo):
    """Draw bounding box, label, and category badge for a person."""
    colour = person.box_color

    # ── Bounding box ───────────────────────────────────────────────────
    thickness = 3 if person.is_worker else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # ── Category badge (small filled circle in top-right corner) ─────────
    if person.is_worker:
        badge_center = (x2 - 12, y1 + 12)
        cv2.circle(frame, badge_center, 11, colour, -1)
        badge_letter = person.badge_text
        cv2.putText(frame, badge_letter, (badge_center[0] - 6, badge_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # ── Label background + text ────────────────────────────────────────
    label = person.display_label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_h = th + 14
    cv2.rectangle(frame, (x1, y1 - label_h), (x1 + tw + 12, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def _draw_hud(frame, width, height, face_id, classifier, banner_text, banner_timer):
    """Draw the heads-up display overlay."""
    # ── Header ─────────────────────────────────────────────────────────
    cv2.putText(frame,
                "Store Intelligence — Role & Category Recognition",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    cv2.putText(frame,
                f"Registered Staff: {len(face_id.known_database)}",
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 2)

    # ── Live counts ────────────────────────────────────────────────────
    counts = classifier.active_counts()
    count_text = (f"Workers: {counts['workers']} (Hackers: {counts['hackers']}, "
                  f"Hustlers: {counts['hustlers']})   Customers: {counts['customers']}")
    cv2.putText(frame, count_text,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 2)

    # ── Legend ─────────────────────────────────────────────────────────
    legend_x = width - 250
    legend_h = 105
    cv2.rectangle(frame, (legend_x, 10), (width - 10, legend_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (legend_x, 10), (width - 10, legend_h), (80, 80, 80), 1)
    cv2.putText(frame, "Legend & Roles", (legend_x + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    y_off = 48
    legend_items = [
        ("Hacker (Lakshana, Mehaa)", ROLE_STYLES["Hacker"]["color"]),
        ("Hustler (Shirley)", ROLE_STYLES["Hustler"]["color"]),
        ("Customer (Shopper)", ROLE_STYLES["Customer"]["color"]),
    ]
    for label_str, color in legend_items:
        cv2.rectangle(frame, (legend_x + 10, y_off - 10),
                      (legend_x + 24, y_off + 2), color, -1)
        cv2.putText(frame, label_str, (legend_x + 30, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        y_off += 18

    # ── Identification banner ──────────────────────────────────────────
    if banner_text and (time.time() - banner_timer) < 4.0:
        cv2.rectangle(frame, (20, height - 70), (width - 20, height - 20),
                      (0, 255, 128), -1)
        cv2.putText(frame, f"  {banner_text}", (30, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(source="0", show_window=True, save_output=False,
                 output_path="sample_data/identified_output.mp4", enable_voice=True):

    video_src = int(source) if source.isdigit() else source

    print(f"🚀 Initializing YOLOv8 & Face Recognition on: {source}")
    model      = YOLO("yolov8n.pt")
    face_id    = FaceIdentifier()
    classifier = PersonClassifier()

    print(f"👥 Known staff registered: {len(face_id.known_database)}")

    # Open video capture (use DirectShow on Windows for reliable webcam access)
    if isinstance(video_src, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(video_src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_src)
    else:
        cap = cv2.VideoCapture(video_src)

    if not cap.isOpened():
        print(f"❌ Cannot open video source '{source}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = None
    if save_output:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Per-track state
    track_box_history = {}                        # track_id -> deque of (cx, cy)
    logged_identities = set()
    banner_text       = ""
    banner_timer      = 0.0
    frame_count       = 0
    consecutive_empty = 0

    window_name = "Store Intelligence — Role & Category Recognition"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n🎥 Running... Press 'q' inside the video window to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_empty += 1
            if consecutive_empty > 20:
                print("❌ Video feed ended or camera disconnected.")
                break
            time.sleep(0.02)
            continue
        consecutive_empty = 0
        frame_count += 1

        # ── Background subtraction (whole frame) ──────────────────────
        fg_mask = bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # ── YOLO detection ─────────────────────────────────────────────
        results = model.track(frame, persist=True,
                               tracker="bytetrack.yaml", classes=[0], verbose=False)

        active_ids = set()

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, track_id, yolo_conf in zip(boxes, track_ids, confs):
                tid = int(track_id)
                active_ids.add(tid)

                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(width, int(box[2])), min(height, int(box[3]))

                # ── Guard 1: Minimum box size ──────────────────────────
                if (x2 - x1) < 40 or (y2 - y1) < 80:
                    continue

                # ── Guard 2: YOLO detection confidence ≥ 0.50 ──────────
                if yolo_conf < 0.50:
                    continue

                cx, cy = _box_centre(x1, y1, x2, y2)

                # ── Guard 3: Background-subtraction motion check ───────
                motion_ratio = _motion_in_mask(fg_mask, x1, y1, x2, y2)
                is_moving = motion_ratio >= 0.03

                # ── Guard 4: Bbox-centre trajectory motion ─────────────
                if tid not in track_box_history:
                    track_box_history[tid] = deque(maxlen=MOTION_FRAMES)
                track_box_history[tid].append((cx, cy))

                traj_motion = _total_path(track_box_history[tid])
                has_trajectory = (len(track_box_history[tid]) >= MOTION_WARMUP_FRAMES
                                  and traj_motion >= MIN_MOTION_PX)

                if not (is_moving or has_trajectory):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                    cv2.putText(frame, "Static (skipped)", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
                    continue

                person_crop = frame[y1:y2, x1:x2]

                # ── Face recognition (every 2 frames) ──────────────────
                face_name = face_id_str = None
                face_conf = 0.0
                person = classifier.get(tid)
                already_known = (person is not None and person.face_name is not None)

                if not already_known and frame_count % 2 == 0:
                    name, pid, conf = face_id.recognize_face_in_crop(
                        person_crop, track_id=tid
                    )
                    if name is not None and conf >= MIN_CONFIDENCE_PCT:
                        face_name   = name
                        face_id_str = pid
                        face_conf   = conf

                # ── Update classification ──────────────────────────────
                person = classifier.update(
                    track_id=tid,
                    face_name=face_name,
                    face_id=face_id_str,
                    face_confidence=face_conf,
                )

                if face_name is not None and face_conf >= MIN_CONFIDENCE_PCT:
                    banner_text  = f"{person.category.upper()} IDENTIFIED: {person.face_name} ({person.face_id})  [{face_conf:.1f}%]"
                    banner_timer = time.time()
                    if enable_voice:
                        announce_voice(person.category, person.face_name)

                # ── Log first identification ───────────────────────────
                log_key = f"{tid}_{person.face_name or 'customer'}"
                if log_key not in logged_identities:
                    logged_identities.add(log_key)
                    if person.is_worker:
                        print(f"🎯 {person.emoji} {person.category}: {person.face_name} (ID: {person.face_id})  "
                              f"Confidence: {person.face_confidence:.1f}%")
                        log_event_to_db("entry",
                                        f"{person.face_name} ({person.face_id}) [{person.category}]",
                                        role=person.category.lower())
                    else:
                        print(f"🎯 🛍️ Customer #{tid} detected")
                        log_event_to_db("entry", f"Customer #{tid}", role="customer")

                # ── Draw ───────────────────────────────────────────────
                _draw_person(frame, x1, y1, x2, y2, person)

        # ── Clean up tracks that disappeared ──────────────────────────
        gone = set(track_box_history) - active_ids
        for tid in gone:
            track_box_history.pop(tid, None)
            classifier.remove(tid)

        # ── HUD overlay ────────────────────────────────────────────────
        _draw_hud(frame, width, height, face_id,
                  classifier, banner_text, banner_timer)

        if out is not None:
            out.write(frame)

        if show_window:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or Esc to quit
                break

    cap.release()
    if out:
        out.release()
        print(f"💾 Saved: {output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store Intelligence pipeline: Category-based Worker (Hacker/Hustler) vs Customer recognition."
    )
    parser.add_argument("--source",   default="0",
                        help="'0' for webcam or path to MP4")
    parser.add_argument("--save",     action="store_true")
    parser.add_argument("--output",   default="sample_data/identified_output.mp4")
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()

    run_pipeline(source=args.source,
                 show_window=not args.no_show,
                 save_output=args.save,
                 output_path=args.output,
                 enable_voice=not args.no_voice)
