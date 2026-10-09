"""
uniform_detector.py
===================
Detects whether a person is wearing a registered uniform.

Three ways to register a uniform:
  1. From an image:       UniformDetector().configure_from_image("uniform.jpg")
  2. From a colour name:  UniformDetector().configure_from_color("blue")
  3. From a description:  UniformDetector().configure_from_description("dark navy polo")

The detector analyses only the **torso region** (middle band of the YOLO
person crop) to avoid confusion with the face or background.

It combines two signals:
  • HSV colour histogram matching  (dominant)
  • Texture energy via Laplacian   (secondary — helps distinguish fabrics
    from skin / printed surfaces)
"""

import os
import sys
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Named-colour table (HSV ranges) ───────────────────────────────────────────
# Each entry: list of (lower_hsv, upper_hsv) tuples.
# Red wraps around 0°/180° in OpenCV HSV so it has two ranges.
NAMED_COLORS: Dict[str, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = {
    "red":    [((0,   90,  60), (12,  255, 255)),
               ((160, 90,  60), (180, 255, 255))],
    "orange": [((11,  100, 80), (24,  255, 255))],
    "yellow": [((25,  80,  80), (35,  255, 255))],
    "green":  [((36,  50,  50), (85,  255, 255))],
    "blue":   [((86,  60,  50), (130, 255, 255))],
    "purple": [((131, 50,  50), (160, 255, 255))],
    "pink":   [((0,   30, 150), (12,  120, 255)),
               ((155, 30, 150), (180, 120, 255))],
    "white":  [((0,    0, 180), (180,  35, 255))],
    "gray":   [((0,    0,  80), (180,  45, 200))],
    "grey":   [((0,    0,  80), (180,  45, 200))],
    "black":  [((0,    0,   0), (180, 255,  70))],
    "brown":  [((5,   60,  40), (18,  200, 170))],
    "cyan":   [((80,  80,  80), (100, 255, 255))],
    "navy":   [((100, 80,  20), (130, 255, 130))],
    "maroon": [((0,   100, 30), (10,  255, 130)),
               ((165, 100, 30), (180, 255, 130))],
    "beige":  [((15,  20, 150), (30,  100, 230))],
    "khaki":  [((18,  30, 100), (35,  130, 220))],
    "teal":   [((80,  80,  60), (95,  255, 200))],
    "olive":  [((25,  40,  40), (45,  200, 170))],
    "cream":  [((18,  10, 190), (35,   60, 255))],
}

# Common colour synonyms / adjectives → base colour
_COLOR_SYNONYMS = {
    "dark blue": "navy",  "light blue": "blue",  "sky blue": "cyan",
    "dark green": "green", "lime": "green",       "forest green": "green",
    "dark red": "maroon",  "crimson": "red",      "scarlet": "red",
    "dark gray": "gray",   "dark grey": "grey",   "light gray": "gray",
    "light grey": "grey",  "charcoal": "gray",    "silver": "gray",
    "golden": "yellow",    "gold": "yellow",      "mustard": "yellow",
    "magenta": "pink",     "violet": "purple",    "indigo": "navy",
    "turquoise": "teal",   "aqua": "cyan",        "tan": "beige",
    "rust": "brown",       "chocolate": "brown",  "burgundy": "maroon",
    "wine": "maroon",      "lavender": "purple",  "peach": "orange",
    "coral": "orange",     "salmon": "pink",      "ivory": "cream",
    "off-white": "cream",  "off white": "cream",
}

CONFIG_PATH = os.path.join("models", "uniform_config.json")


class UniformDetector:
    """
    Classifies each detected person as WORKER or VISITOR based on
    whether their torso matches the registered uniform colour profile.

    Supports multiple uniform profiles (e.g. different departments).
    """

    def __init__(self, config_path: str = CONFIG_PATH, match_threshold: float = 0.18):
        """
        Parameters
        ----------
        config_path     : path to the JSON file that stores the uniform profile.
        match_threshold : fraction of torso pixels that must match the uniform
                          colour for a person to be classified as a Worker.
                          Default 0.18 (18 %).
        """
        self.config_path     = config_path
        self.match_threshold = match_threshold
        self._profiles: List[dict] = []   # list of uniform profiles
        self._loaded            = False
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        self._try_load()

    # ── Configuration ──────────────────────────────────────────────────────────

    def configure_from_color(self, color_name: str) -> bool:
        """Register a uniform by colour name (e.g. 'blue', 'red', 'green')."""
        key = color_name.strip().lower()
        if key not in NAMED_COLORS:
            available = ", ".join(sorted(NAMED_COLORS))
            print(f"❌ Unknown colour '{color_name}'. Available: {available}")
            return False

        ranges = NAMED_COLORS[key]
        hsv_ranges = [(list(lo), list(hi)) for lo, hi in ranges]
        label = color_name.capitalize()

        self._add_profile(label=label, hsv_ranges=hsv_ranges,
                          source="color", description=f"Colour: {label}")
        print(f"✅ Uniform registered as colour: {label}")
        return True

    def configure_from_description(self, description: str) -> bool:
        """
        Register a uniform from a free-text description.
        Parses known colour words from the text to build a colour profile.

        Examples
        --------
        >>> det.configure_from_description("dark navy polo shirt")
        >>> det.configure_from_description("red and white striped t-shirt")
        """
        desc_lower = description.strip().lower()
        found_colors: List[str] = []
        found_ranges: list = []

        # First check multi-word synonyms (longest match first)
        for synonym, base_color in sorted(_COLOR_SYNONYMS.items(),
                                          key=lambda x: -len(x[0])):
            if synonym in desc_lower:
                if base_color not in found_colors:
                    found_colors.append(base_color)
                    for lo, hi in NAMED_COLORS.get(base_color, []):
                        found_ranges.append((list(lo), list(hi)))
                # Remove the synonym so its constituent words aren't matched again
                desc_lower = desc_lower.replace(synonym, " ")

        # Then check single colour names
        for color_name, ranges in NAMED_COLORS.items():
            if color_name in desc_lower.split() and color_name not in found_colors:
                found_colors.append(color_name)
                for lo, hi in ranges:
                    found_ranges.append((list(lo), list(hi)))

        if not found_ranges:
            available = ", ".join(sorted(NAMED_COLORS))
            print(f"❌ No recognised colours in description: '{description}'")
            print(f"   Known colours: {available}")
            print(f"   Known synonyms: {', '.join(sorted(_COLOR_SYNONYMS))}")
            return False

        label = " + ".join(c.capitalize() for c in found_colors)
        self._add_profile(label=label, hsv_ranges=found_ranges,
                          source="description", description=description)
        print(f"✅ Uniform registered from description: \"{description}\"")
        print(f"   Detected colours: {label}")
        return True

    def configure_from_image(self, image_path: str, n_clusters: int = 5) -> bool:
        """
        Register a uniform by extracting the dominant torso colour from
        a reference image (photo of a staff member or a fabric swatch).
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return False

        torso = self._crop_torso(img)
        hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # Flatten pixels for k-means clustering
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Exclude very dark/near-black pixels (likely background)
        bright_mask = pixels[:, 2] > 40
        pixels = pixels[bright_mask]
        if len(pixels) < 50:
            print("❌ Not enough bright pixels in the image — try a brighter photo.")
            return False

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centres = cv2.kmeans(
            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Pick the cluster with the most pixels
        counts = np.bincount(labels.flatten())
        dominant_idx = int(np.argmax(counts))
        dom_hsv = centres[dominant_idx]  # (H, S, V) as float

        # Build a tolerance range around the dominant hue
        h_tol = 18   # ± 18° on the hue wheel
        s_tol = 60
        v_tol = 70

        lo = [max(0,   int(dom_hsv[0]) - h_tol),
              max(0,   int(dom_hsv[1]) - s_tol),
              max(0,   int(dom_hsv[2]) - v_tol)]
        hi = [min(180, int(dom_hsv[0]) + h_tol),
              min(255, int(dom_hsv[1]) + s_tol),
              min(255, int(dom_hsv[2]) + v_tol)]

        # Handle red hue wrap-around
        ranges = [(lo, hi)]
        if lo[0] < 10:
            # Also add the wrap-around range near 180
            ranges.append(([180 + lo[0] - 10, lo[1], lo[2]],
                           [180,               hi[1], hi[2]]))

        # Compute a texture fingerprint from the reference image
        texture_energy = self._compute_texture_energy(torso)

        label = f"Image({os.path.basename(image_path)})"
        self._add_profile(label=label, hsv_ranges=ranges,
                          source="image", description=f"From: {image_path}",
                          texture_energy=texture_energy)
        print(f"✅ Uniform registered from image: {image_path}")
        print(f"   Dominant HSV: H={dom_hsv[0]:.0f}° S={dom_hsv[1]:.0f} V={dom_hsv[2]:.0f}")
        print(f"   Match range : {lo} → {hi}")
        print(f"   Texture energy: {texture_energy:.1f}")
        return True

    def clear_profiles(self):
        """Remove all registered uniform profiles."""
        self._profiles = []
        self._save()
        print("🗑️  All uniform profiles cleared.")

    # ── Classification ─────────────────────────────────────────────────────────

    def classify(self, person_crop_bgr) -> dict:
        """
        Full classification returning a structured result.

        Returns
        -------
        dict with keys:
            role          : "worker" | "visitor"
            confidence    : float 0–100 (match confidence %)
            match_ratio   : float 0–100 (% of torso pixels matching)
            profile_label : str (which profile matched, or "")
        """
        if not self._loaded or not self._profiles:
            return {"role": "visitor", "confidence": 0.0,
                    "match_ratio": 0.0, "profile_label": ""}
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return {"role": "visitor", "confidence": 0.0,
                    "match_ratio": 0.0, "profile_label": ""}

        torso = self._crop_torso(person_crop_bgr)
        if torso.size == 0:
            return {"role": "visitor", "confidence": 0.0,
                    "match_ratio": 0.0, "profile_label": ""}

        hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]

        best_ratio   = 0.0
        best_label   = ""
        best_texture = 0.0

        for profile in self._profiles:
            hsv_ranges = profile.get("hsv_ranges", [])
            if not hsv_ranges:
                continue

            # Colour match
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in hsv_ranges:
                lo_np = np.array(lo, dtype=np.uint8)
                hi_np = np.array(hi, dtype=np.uint8)
                combined_mask = cv2.bitwise_or(
                    combined_mask, cv2.inRange(hsv, lo_np, hi_np)
                )
            match_ratio = float(np.count_nonzero(combined_mask)) / max(total, 1)

            # Texture similarity bonus (if reference texture was captured)
            texture_bonus = 0.0
            ref_texture = profile.get("texture_energy")
            if ref_texture and ref_texture > 0:
                curr_texture = self._compute_texture_energy(torso)
                # Similarity: 1.0 when identical, decays as they diverge
                ratio = min(curr_texture, ref_texture) / max(curr_texture, ref_texture, 1)
                texture_bonus = ratio * 0.05  # up to 5% bonus

            effective_ratio = match_ratio + texture_bonus

            if effective_ratio > best_ratio:
                best_ratio   = effective_ratio
                best_label   = profile.get("label", "Unknown")
                best_texture = texture_bonus

        is_worker = best_ratio >= self.match_threshold
        # Confidence: how far above threshold (capped at 100)
        if is_worker:
            confidence = min(100.0, (best_ratio / max(self.match_threshold, 0.01)) * 50 + 50)
        else:
            confidence = max(0.0, (best_ratio / max(self.match_threshold, 0.01)) * 50)

        return {
            "role":          "worker" if is_worker else "visitor",
            "confidence":    round(confidence, 1),
            "match_ratio":   round(best_ratio * 100, 1),
            "profile_label": best_label if is_worker else "",
        }

    def is_worker(self, person_crop_bgr) -> tuple:
        """
        Legacy API — returns (is_worker: bool, match_ratio: float).
        match_ratio is the percentage of torso pixels matching the uniform colour.
        """
        result = self.classify(person_crop_bgr)
        return result["role"] == "worker", result["match_ratio"]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _crop_torso(self, crop_bgr):
        """
        Return the torso region of a person crop.
        We take the horizontal band from 25 % to 65 % of the height
        (just below the neck, above the hips) and 15 %–85 % of the width.
        This avoids the face, background, and legs.
        """
        h, w = crop_bgr.shape[:2]
        y1 = max(0, int(h * 0.25))
        y2 = max(y1 + 5, int(h * 0.65))
        x1 = max(0, int(w * 0.15))
        x2 = max(x1 + 5, int(w * 0.85))
        return crop_bgr[y1:y2, x1:x2]

    def _compute_texture_energy(self, region_bgr) -> float:
        """Compute a scalar texture-energy measure (Laplacian variance)."""
        if region_bgr is None or region_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    def _add_profile(self, *, label: str, hsv_ranges: list,
                     source: str, description: str,
                     texture_energy: Optional[float] = None):
        """Add a uniform profile and save to disk."""
        profile = {
            "label":          label,
            "hsv_ranges":     hsv_ranges,
            "source":         source,
            "description":    description,
        }
        if texture_energy is not None:
            profile["texture_energy"] = round(texture_energy, 2)

        # Replace existing profile from same source+label, or append
        replaced = False
        for i, p in enumerate(self._profiles):
            if p.get("label") == label and p.get("source") == source:
                self._profiles[i] = profile
                replaced = True
                break
        if not replaced:
            self._profiles.append(profile)

        self._save()

    def _save(self):
        data = {
            "profiles":        self._profiles,
            "match_threshold": self.match_threshold,
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)
        self._loaded = bool(self._profiles)

    def _try_load(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path) as f:
                data = json.load(f)

            # Support both old (single-profile) and new (multi-profile) format
            if "profiles" in data:
                self._profiles = data["profiles"]
            elif "hsv_ranges" in data:
                # Migrate old single-profile config
                self._profiles = [{
                    "label":       data.get("label", "Legacy"),
                    "hsv_ranges":  data["hsv_ranges"],
                    "source":      "color",
                    "description": f"Migrated: {data.get('label', '')}",
                }]
            else:
                self._profiles = []

            self.match_threshold = data.get("match_threshold", self.match_threshold)
            self._loaded = bool(self._profiles)
            if self._loaded:
                labels = [p.get("label", "?") for p in self._profiles]
                print(f"✅ Uniform profile(s) loaded: {', '.join(labels)}")
        except Exception as e:
            print(f"⚠️  Could not load uniform config: {e}")

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return self._loaded

    @property
    def label(self) -> str:
        if not self._profiles:
            return "Not configured"
        return " | ".join(p.get("label", "?") for p in self._profiles)

    @property
    def profile_count(self) -> int:
        return len(self._profiles)

    def show_color_preview(self, window_name="Uniform Colour"):
        """Displays a colour swatch of the registered uniform colour."""
        if not self._loaded:
            print("No uniform configured yet.")
            return
        swatch = np.zeros((100, 400, 3), dtype=np.uint8)
        for profile in self._profiles:
            for lo, hi in profile.get("hsv_ranges", []):
                mid_h = (lo[0] + hi[0]) // 2
                mid_s = min(255, (lo[1] + hi[1]) // 2 + 60)
                mid_v = min(255, (lo[2] + hi[2]) // 2 + 60)
                hsv_swatch = np.full((100, 400, 3), [mid_h, mid_s, mid_v], dtype=np.uint8)
                bgr_swatch = cv2.cvtColor(hsv_swatch, cv2.COLOR_HSV2BGR)
                swatch = cv2.addWeighted(swatch, 0.3, bgr_swatch, 0.7, 0)

        label_text = self.label
        if len(label_text) > 45:
            label_text = label_text[:42] + "..."
        cv2.putText(swatch, f"Uniform: {label_text}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        try:
            cv2.imshow(window_name, swatch)
            cv2.waitKey(2000)
            cv2.destroyWindow(window_name)
        except Exception:
            pass
