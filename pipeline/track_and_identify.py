"""
track_and_identify.py
=====================
Main video pipeline:  YOLO detection → ByteTrack → Face ID + Uniform → Role
classification.

Each detected person is classified into one of four roles:
  • Known Worker   — recognised face + wearing uniform
  • Unknown Worker — unrecognised face + wearing uniform
  • Known Visitor  — recognised face + NOT wearing uniform
  • Visitor        — unrecognised face + NOT wearing uniform
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
from uniform_detector import UniformDetector
from person_classifier import (
    PersonClassifier, PersonInfo,
    ROLE_KNOWN_WORKER, ROLE_UNKNOWN_WORKER, ROLE_KNOWN_VISITOR, ROLE_VISITOR,
    ROLE_STYLES,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# Minimum confidence (%) to accept a face match as a real identification
MIN_CONFIDENCE_PCT = 85.0

# Motion tracker: how many frames to observe & minimum cumulative bbox movement
MOTION_FRAMES       = 12   # observe over 12 frames
MIN_MOTION_PX       = 8.0  # a real moving person's box centre moves ≥ 8 px total
# We wait until we have accumulated enough motion history before allowing recognition
MOTION_WARMUP_FRAMES = MOTION_FRAMES

# Background subtractor for coarse "is something actually moving here?" check
# This is the strongest defence against static paintings/posters
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120, varThreshold=40, detectShadows=False
)

# ── Voice ──────────────────────────────────────────────────────────────────────
last_speech_time = {}

def announce_voice(name):
    now = time.time()
    if name not in last_speech_time or (now - last_speech_time[name]) > 8.0:
        last_speech_time[name] = now
        try:
            cmd = (f'Add-Type -AssemblyName System.Speech; '
                   f'(New-Object System.Speech.Synthesis.SpeechSynthesizer)'
                   f'.Speak("Identified {name}")')
            subprocess.Popen(['powershell', '-Command', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

# ── Database ───────────────────────────────────────────────────────────────────
def log_event_to_db(event_type, person_id, zone="store", role="visitor"):
    try:
        conn = sqlite3.connect("store.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT, person_id TEXT,
                zone TEXT, role TEXT DEFAULT 'visitor',
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

# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_role_badge(frame, x1, y1, x2, person: PersonInfo):
    """Draw the bounding box + label badge for a classified person."""
    colour = person.box_color
    label  = person.display_label

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, person.track_id), colour, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y1), colour, 2)  # reset

    # Full bbox
    cv2.rectangle(frame, (x1, y1), (x2, int(frame.shape[0])), colour, 2)
    # Just draw bbox properly
    pass


def _draw_person(frame, x1, y1, x2, y2, person: PersonInfo):
    """Draw bounding box, label, and confidence bar for a person."""
    colour = person.box_color

    # ── Bounding box ───────────────────────────────────────────────────
    thickness = 3 if person.is_worker else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    # ── Worker badge (small filled circle in top-right corner) ─────────
    if person.is_worker:
        badge_center = (x2 - 12, y1 + 12)
        cv2.circle(frame, badge_center, 10, colour, -1)
        cv2.putText(frame, "W", (badge_center[0] - 6, badge_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

    # ── Label background + text ────────────────────────────────────────
    label = person.display_label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_h = th + 14
    cv2.rectangle(frame, (x1, y1 - label_h), (x1 + tw + 12, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # ── Uniform confidence bar (below label, only if uniform is active) ─
    if person.uniform_ratio > 0:
        bar_w = int((x2 - x1) * min(person.uniform_ratio / 100, 1.0))
        bar_y = y2 + 4
        bar_colour = (0, 200, 0) if person.uniform_match else (80, 80, 80)
        cv2.rectangle(frame, (x1, bar_y), (x1 + bar_w, bar_y + 5), bar_colour, -1)
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + 5), (100, 100, 100), 1)


def _draw_hud(frame, width, height, uniform_dt, face_id, classifier, banner_text, banner_timer):
    """Draw the heads-up display overlay."""
    # ── Header ─────────────────────────────────────────────────────────
    cv2.putText(frame,
                "Store Intelligence — YOLO + Face ID + Uniform Detection",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    uniform_status = uniform_dt.label if uniform_dt.is_configured else "NOT SET"
    cv2.putText(frame,
                f"Uniform: {uniform_status}   Known faces: {len(face_id.known_database)}",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 220), 2)

    # ── Live counts ────────────────────────────────────────────────────
    counts = classifier.active_counts()
    count_text = (f"Workers: {counts[ROLE_KNOWN_WORKER] + counts[ROLE_UNKNOWN_WORKER]}  "
                  f"Visitors: {counts[ROLE_KNOWN_VISITOR] + counts[ROLE_VISITOR]}")
    cv2.putText(frame, count_text,
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 2)

    # ── Legend ─────────────────────────────────────────────────────────
    legend_x = width - 240
    cv2.rectangle(frame, (legend_x, 10), (width - 10, 120), (30, 30, 30), -1)
    cv2.rectangle(frame, (legend_x, 10), (width - 10, 120), (80, 80, 80), 1)
    cv2.putText(frame, "Legend", (legend_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y_off = 48
    for role_key, style in ROLE_STYLES.items():
        role_label = style["prefix"]
        role_color = style["color"]
        cv2.rectangle(frame, (legend_x + 10, y_off - 10),
                      (legend_x + 26, y_off + 2), role_color, -1)
        cv2.putText(frame, role_label, (legend_x + 32, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        y_off += 18

    # ── Identification banner ──────────────────────────────────────────
    if banner_text and (time.time() - banner_timer) < 4.0:
        cv2.rectangle(frame, (20, height - 70), (width - 20, height - 20),
                      (0, 200, 0), -1)
        cv2.putText(frame, f"  {banner_text}", (30, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2)


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(source="0", show_window=True, save_output=False,
                 output_path="sample_data/identified_output.mp4", enable_voice=True):

    video_src = int(source) if source.isdigit() else source

    print(f"🚀 Initializing YOLOv8 & Face Recognition on: {source}")
    model      = YOLO("yolov8n.pt")
    face_id    = FaceIdentifier()
    uniform_dt = UniformDetector()
    classifier = PersonClassifier(confirm_frames=5)

    if uniform_dt.is_configured:
        print(f"👕 Uniform profile loaded: {uniform_dt.label}")
    else:
        print("⚠️  No uniform profile configured. Run:")
        print("   python pipeline/register_uniform.py --color <colour>")
        print("   python pipeline/register_uniform.py --describe \"navy blue polo\"")
        print("   python pipeline/register_uniform.py --image <photo.jpg>")
        print("   Everyone will be shown as VISITOR until a uniform is registered.")

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"❌ Cannot open video source '{source}'")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    print("\n🎥 Running... Press 'q' to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── Background subtraction (whole frame) ──────────────────────
        fg_mask = bg_subtractor.apply(frame)
        # Small morphological clean-up to remove noise
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

                # ── Guard 2: YOLO detection confidence ≥ 0.5 ──────────
                if yolo_conf < 0.50:
                    continue

                cx, cy = _box_centre(x1, y1, x2, y2)

                # ── Guard 3: Background-subtraction motion check ───────
                # If < 3 % of the person's bbox area is moving pixels,
                # this is almost certainly a static object (painting/poster).
                motion_ratio = _motion_in_mask(fg_mask, x1, y1, x2, y2)
                is_moving = motion_ratio >= 0.03

                # ── Guard 4: Bbox-centre trajectory motion ─────────────
                if tid not in track_box_history:
                    track_box_history[tid] = deque(maxlen=MOTION_FRAMES)
                track_box_history[tid].append((cx, cy))

                traj_motion = _total_path(track_box_history[tid])
                has_trajectory = (len(track_box_history[tid]) >= MOTION_WARMUP_FRAMES
                                  and traj_motion >= MIN_MOTION_PX)

                # A track passes if it has moving pixels OR has moved over the window
                if not (is_moving or has_trajectory):
                    # Draw a red "static / not a person" box and skip recognition
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                    cv2.putText(frame, "Static (skipped)", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
                    continue

                person_crop = frame[y1:y2, x1:x2]

                # ── Face recognition (every 3 frames) ──────────────────
                face_name = face_id_str = None
                face_conf = 0.0
                person = classifier.get(tid)
                already_known = (person is not None and person.face_name is not None)

                if not already_known and frame_count % 3 == 0:
                    name, pid, conf = face_id.recognize_face_in_crop(
                        person_crop, track_id=tid
                    )
                    if name is not None and conf >= MIN_CONFIDENCE_PCT:
                        face_name  = name
                        face_id_str = pid
                        face_conf  = conf

                        banner_text  = f"IDENTIFIED: {name} ({pid})  [{conf:.1f}%]"
                        banner_timer = time.time()

                        if enable_voice:
                            announce_voice(name)

                # ── Uniform detection (every 2 frames) ─────────────────
                uniform_result = None
                if uniform_dt.is_configured and frame_count % 2 == 0:
                    uniform_result = uniform_dt.classify(person_crop)

                # ── Fuse into final classification ─────────────────────
                person = classifier.update(
                    track_id=tid,
                    face_name=face_name,
                    face_id=face_id_str,
                    face_confidence=face_conf,
                    uniform_result=uniform_result,
                )

                # ── Log first identification ───────────────────────────
                log_key = f"{tid}_{person.face_name or 'anon'}"
                if log_key not in logged_identities:
                    if person.face_name:
                        logged_identities.add(log_key)
                        role_emoji = "👷" if person.is_worker else "🧑"
                        print(f"🎯 {role_emoji} {person.face_name} (ID: {person.face_id})  "
                              f"Role: {person.role}  Confidence: {person.face_confidence:.1f}%")
                        log_event_to_db("entry",
                                        f"{person.face_name} ({person.face_id})",
                                        role=person.role)

                # ── Draw ───────────────────────────────────────────────
                _draw_person(frame, x1, y1, x2, y2, person)

        # ── Clean up tracks that disappeared ──────────────────────────
        gone = set(track_box_history) - active_ids
        for tid in gone:
            track_box_history.pop(tid, None)
            classifier.remove(tid)

        # ── HUD overlay ────────────────────────────────────────────────
        _draw_hud(frame, width, height, uniform_dt, face_id,
                  classifier, banner_text, banner_timer)

        if out is not None:
            out.write(frame)

        if show_window:
            cv2.imshow("Store Intelligence — Worker / Visitor Classification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
        print(f"💾 Saved: {output_path}")
    cv2.destroyAllWindows()
    print("✅ Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store Intelligence pipeline with Worker/Visitor classification."
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
