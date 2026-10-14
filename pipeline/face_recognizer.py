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

    def _check_face_liveness(self, face, crop_bgr, crop_h, crop_w, track_id=None):
        """
        Multi-factor liveness and 2D painting check.
        Returns: (is_live: bool, is_painting: bool)
        """
        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

        # ── 1. Basic size & position guard ────────────────────────────
        if fw < self.min_face_w or fh < self.min_face_h:
            return False, False
        face_center_y = fy + fh / 2
        if crop_h > 0 and (face_center_y / crop_h) > 0.65:
            return False, False

        aspect = fw / max(fh, 1)
        if aspect < 0.45 or aspect > 1.65:
            return False, True  # Unnatural distorted painted face

        # ── 2. Skin-tone check (HSV range for human skin) ─────────────
        x1 = max(0, fx); y1 = max(0, fy)
        x2 = min(crop_w, fx + fw); y2 = min(crop_h, fy + fh)
        face_region = crop_bgr[y1:y2, x1:x2]
        if face_region.size == 0:
            return False, False

        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 20, 60), (25, 175, 255))
        mask2 = cv2.inRange(hsv, (165, 20, 60), (180, 175, 255))
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_ratio = np.count_nonzero(skin_mask) / max(skin_mask.size, 1)

        # Paintings often use artificial oil/acrylic pigments
        if skin_ratio < 0.12:
            return False, True

        # ── 3. Eye-region texture (Laplacian variance) ────────────────
        try:
            rex, rey = int(face[4]), int(face[5])
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
                return float(lap.var())

            right_var = eye_patch_variance(rex, rey)
            left_var  = eye_patch_variance(lex, ley)
            avg_eye_var = (right_var + left_var) / 2

            # Paintings have smooth brush strokes/prints (< 65 var)
            # Real human eyes with specular highlights & lashes are > 65
            if avg_eye_var < 65.0:
                return False, True

            # ── 4. Landmark geometry sanity ───────────────────────────
            nose_x, nose_y = int(face[8]), int(face[9])
            inter_eye_dist = np.hypot(rex - lex, rey - ley)
            if inter_eye_dist < 6:
                return False, True
            if nose_y <= min(rey, ley):
                return False, True

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
                    diffs = np.diff(positions, axis=0)
                    total_motion = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
                    if total_motion < self._MIN_MOTION_PX:
                        return False, True  # Perfectly static on wall -> painting!

        except (IndexError, TypeError):
            pass

        return True, False

    def is_artwork_crop(self, crop_bgr):
        """
        Detects if a bounding box crop is a wall painting, poster, or framed picture.
        Checks for:
          1. Rectangular frame borders along the perimeter
          2. Abnormally flat/uniform texture or low Laplacian variance
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        h, w = crop_bgr.shape[:2]
        if h < 40 or w < 30:
            return False

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Check for prominent rectangular frame border lines along crop edges
        border_t = max(2, int(h * 0.10))
        border_s = max(2, int(w * 0.10))
        if border_t > 2 and border_s > 2:
            edges = cv2.Canny(gray, 60, 160)
            top_strip    = edges[:border_t, :]
            bottom_strip = edges[h - border_t:, :]
            left_strip   = edges[:, :border_s]
            right_strip  = edges[:, w - border_s:]

            top_density    = np.count_nonzero(top_strip) / max(top_strip.size, 1)
            bottom_density = np.count_nonzero(bottom_strip) / max(bottom_strip.size, 1)
            left_density   = np.count_nonzero(left_strip) / max(left_strip.size, 1)
            right_density  = np.count_nonzero(right_strip) / max(right_strip.size, 1)

            # A framed painting has distinct border lines along top/bottom AND left/right
            if (top_density > 0.08 and bottom_density > 0.08) and (left_density > 0.08 and right_density > 0.08):
                return True

        # 2. Texture flatness check: drawings/posters have unnaturally flat surface texture
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 18.0:
            return True

        return False

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
        Returns: (name, person_id, confidence%, is_painting: bool)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None, None, 0.0, False

        h, w = crop_bgr.shape[:2]
        if h < 50 or w < 35:
            return None, None, 0.0, False

        # Visual artwork surface test
        is_art = self.is_artwork_crop(crop_bgr)

        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(crop_bgr)
        has_faces = faces is not None and len(faces) > 0

        if not has_faces:
            # No face detected by YuNet. If artwork features matched, flag as painting
            return None, None, 0.0, is_art

        # Evaluate face liveness
        live_faces = []
        is_painting_face = False
        for f in faces:
            is_live, is_fake = self._check_face_liveness(f, crop_bgr, h, w, track_id)
            if is_live:
                live_faces.append(f)
            elif is_fake:
                is_painting_face = True

        if not live_faces:
            # Face was detected, but failed liveness -> 2D painting / poster!
            return None, None, 0.0, True

        # Process live human face
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
            return best_match, best_id, conf, False

        # Real living human face that is not in staff database (customer)
        return None, None, 0.0, False
