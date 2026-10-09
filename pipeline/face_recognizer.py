import os
import cv2
import numpy as np
from collections import deque

class FaceIdentifier:
    def __init__(self, 
                 yunet_model="models/face_detection_yunet_2023mar.onnx",
                 sface_model="models/face_recognition_sface_2021dec.onnx",
                 known_dir="known_people",
                 cosine_threshold=0.40):
        
        self.known_dir = known_dir
        self.cosine_threshold = cosine_threshold
        os.makedirs(self.known_dir, exist_ok=True)
        
        # YuNet with high threshold
        self.detector = cv2.FaceDetectorYN.create(
            yunet_model, "", (320, 320), score_threshold=0.85
        )
        self.recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
        
        # Minimum face pixel dimensions
        self.min_face_w = 45
        self.min_face_h = 45

        # Track face crops over frames to measure micro-movement (liveness)
        # Maps track_id -> deque of previous eye-center positions
        self._motion_history = {}  # track_id -> deque of (eye_cx, eye_cy)
        self._liveness_cache = {}  # track_id -> (is_live, frame_count)
        self._MOTION_WINDOW = 8    # frames to observe for motion
        self._MIN_MOTION_PX = 1.5  # minimum pixel shift over window to be "live"

        self.known_database = {}
        self.load_known_people()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_real_human_face(self, face, crop_bgr, crop_h, crop_w, track_id=None):
        """
        Multi-factor liveness check.
        Returns True only if this looks like a real 3D human face.
        """
        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        
        # ── 1. Basic size & position guard ────────────────────────────
        if fw < self.min_face_w or fh < self.min_face_h:
            return False
        face_center_y = fy + fh / 2
        if crop_h > 0 and (face_center_y / crop_h) > 0.60:
            return False  # Face in lower 40 % of body crop → likely not a face

        # Aspect ratio: real faces are roughly square to slightly tall
        aspect = fw / max(fh, 1)
        if aspect < 0.5 or aspect > 1.6:
            return False

        # Face must cover at least 2 % of the person crop area
        if (fw * fh) / max(crop_h * crop_w, 1) < 0.02:
            return False

        # ── 2. Skin-tone check (HSV range for human skin) ─────────────
        x1 = max(0, fx); y1 = max(0, fy)
        x2 = min(crop_w, fx + fw); y2 = min(crop_h, fy + fh)
        face_region = crop_bgr[y1:y2, x1:x2]
        if face_region.size == 0:
            return False
        
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        # Human skin HSV: H 0-25 or 165-180, S 20-175, V 60-255
        mask1 = cv2.inRange(hsv, (0, 20, 60), (25, 175, 255))
        mask2 = cv2.inRange(hsv, (165, 20, 60), (180, 175, 255))
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_ratio = np.count_nonzero(skin_mask) / max(skin_mask.size, 1)
        
        # A real face crop should have at least 10 % skin-tone pixels
        if skin_ratio < 0.10:
            return False

        # ── 3. Eye-region texture (Laplacian variance) ────────────────
        # Paintings have very smooth eye areas; real eyes have fine texture
        # YuNet provides 5 landmarks: [right_eye, left_eye, nose, r_mouth, l_mouth]
        # Landmarks start at index 4 in the face array: (x4,y4), (x5,y5), ...
        try:
            # Right eye
            rex, rey = int(face[4]), int(face[5])
            # Left eye
            lex, ley = int(face[6]), int(face[7])
            
            eye_patch_size = max(10, fw // 5)
            
            def eye_patch_variance(cx, cy):
                ex1 = max(0, cx - eye_patch_size)
                ey1 = max(0, cy - eye_patch_size)
                ex2 = min(crop_w, cx + eye_patch_size)
                ey2 = min(crop_h, cy + eye_patch_size)
                patch = crop_bgr[ey1:ey2, ex1:ex2]
                if patch.size == 0:
                    return 999.0
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                return lap.var()
            
            right_var = eye_patch_variance(rex, rey)
            left_var  = eye_patch_variance(lex, ley)
            avg_eye_var = (right_var + left_var) / 2
            
            # Flat/painted eyes have very low Laplacian variance (< 60)
            # Real eyes (with texture, reflections, lashes) are typically > 60
            if avg_eye_var < 60.0:
                return False
            
            # ── 4. Landmark geometry sanity ───────────────────────────
            nose_x, nose_y = int(face[8]), int(face[9])
            
            inter_eye_dist = np.hypot(rex - lex, rey - ley)
            if inter_eye_dist < 5:
                return False  # Eyes on top of each other → bad detection
            
            # Nose should be below both eyes
            if nose_y <= min(rey, ley):
                return False
            
            # Nose should be roughly centred between the eyes
            nose_x_ratio = abs(nose_x - (rex + lex) / 2) / max(inter_eye_dist, 1)
            if nose_x_ratio > 0.7:
                return False  # Nose too far to one side
            
            # ── 5. Micro-motion liveness (over several frames) ────────
            if track_id is not None:
                eye_cx = (rex + lex) / 2
                eye_cy = (rey + ley) / 2
                
                if track_id not in self._motion_history:
                    self._motion_history[track_id] = deque(maxlen=self._MOTION_WINDOW)
                
                self._motion_history[track_id].append((eye_cx, eye_cy))
                history = self._motion_history[track_id]
                
                if len(history) >= self._MOTION_WINDOW:
                    positions = np.array(list(history))
                    # Total path length over the observation window
                    diffs = np.diff(positions, axis=0)
                    total_motion = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
                    
                    # Paintings: total_motion ≈ 0 (perfectly static)
                    # Real faces: even if the person is still, camera noise
                    # and micro-expressions create > 1.5 px of total movement
                    if total_motion < self._MIN_MOTION_PX:
                        return False  # Too static → likely a painting/poster

        except (IndexError, TypeError):
            pass  # If landmarks are malformed, skip geometry checks

        return True

    def _extract_face_feature(self, img_bgr):
        """Extract embedding from best face in a reference photo."""
        h, w, _ = img_bgr.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None
        best_face = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(img_bgr, best_face)
        return self.recognizer.feature(aligned)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_known_people(self):
        """Loads and computes embeddings for all images in known_people dir."""
        self.known_database = {}
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if not os.path.exists(self.known_dir):
            return
        for file in os.listdir(self.known_dir):
            if file.lower().endswith(valid_ext):
                base = os.path.splitext(file)[0]
                if "_" in base:
                    name, pid = base.split("_", 1)
                else:
                    name = base
                    pid = f"ID-{base.upper()[:4]}"
                img = cv2.imread(os.path.join(self.known_dir, file))
                if img is None:
                    continue
                feat = self._extract_face_feature(img)
                if feat is not None:
                    self.known_database[name] = {
                        "name": name, "person_id": pid, "feature": feat
                    }
                    print(f"✅ Registered: {name} ({pid}) from {file}")
                else:
                    print(f"⚠️  No face in reference photo: {file}")

    def recognize_face_in_crop(self, crop_bgr, track_id=None):
        """
        Identify a face within a YOLO person bounding box.
        track_id is passed to enable cross-frame micro-motion liveness.
        Returns (name, person_id, confidence%) or (None, None, 0.0).
        """
        if not self.known_database:
            return None, None, 0.0
        if crop_bgr is None or crop_bgr.size == 0:
            return None, None, 0.0

        h, w = crop_bgr.shape[:2]
        if h < 60 or w < 40:
            return None, None, 0.0

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(crop_bgr)
        if faces is None or len(faces) == 0:
            return None, None, 0.0

        # Apply full liveness filter
        live_faces = [
            f for f in faces
            if self._is_real_human_face(f, crop_bgr, h, w, track_id)
        ]
        if not live_faces:
            return None, None, 0.0

        best_face = max(live_faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(crop_bgr, best_face)
        feature = self.recognizer.feature(aligned)

        best_match = best_id = None
        best_score = -1.0
        for name, data in self.known_database.items():
            score = self.recognizer.match(
                feature, data["feature"], cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > self.cosine_threshold and score > best_score:
                best_score = score
                best_match = name
                best_id = data["person_id"]

        if best_match is not None:
            conf = min(100.0, max(0.0, (best_score / 0.8) * 100))
            return best_match, best_id, conf

        return None, None, 0.0
