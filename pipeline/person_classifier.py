"""
person_classifier.py
====================
High-level classifier that fuses face recognition and uniform detection
to produce a single role label for each tracked person:

    ┌──────────────────────┬────────────────┬───────────────────────┐
    │                      │ Uniform match  │ No uniform match      │
    ├──────────────────────┼────────────────┼───────────────────────┤
    │ Face recognised      │ KNOWN WORKER   │ KNOWN VISITOR         │
    │ Face NOT recognised  │ UNKNOWN WORKER │ VISITOR               │
    └──────────────────────┴────────────────┴───────────────────────┘

This module is imported by track_and_identify.py for the live pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Role constants ─────────────────────────────────────────────────────────────
ROLE_KNOWN_WORKER   = "known_worker"
ROLE_UNKNOWN_WORKER = "unknown_worker"
ROLE_KNOWN_VISITOR  = "known_visitor"
ROLE_VISITOR        = "visitor"

# Visual styling for each role (BGR colour, display label prefix, emoji)
ROLE_STYLES = {
    ROLE_KNOWN_WORKER:   {"color": (0, 200, 255), "prefix": "★ Worker",  "emoji": "👷"},
    ROLE_UNKNOWN_WORKER: {"color": (255, 50, 50), "prefix": "Worker",    "emoji": "🦺"},
    ROLE_KNOWN_VISITOR:  {"color": (0, 220, 0),   "prefix": "★ Visitor", "emoji": "🧑"},
    ROLE_VISITOR:        {"color": (255, 140, 0),  "prefix": "Visitor",   "emoji": "👤"},
}


@dataclass
class PersonInfo:
    """Everything we know about a tracked person in the current session."""
    track_id:          int
    role:              str   = ROLE_VISITOR
    # Face recognition
    face_name:         Optional[str] = None
    face_id:           Optional[str] = None
    face_confidence:   float = 0.0
    # Uniform detection
    uniform_match:     bool  = False
    uniform_ratio:     float = 0.0
    uniform_label:     str   = ""
    uniform_confidence: float = 0.0
    # Smoothing: how many consecutive frames supported each signal
    _uniform_streak:   int   = field(default=0, repr=False)
    _visitor_streak:   int   = field(default=0, repr=False)

    # ── Display helpers ────────────────────────────────────────────────────

    @property
    def display_label(self) -> str:
        style = ROLE_STYLES.get(self.role, ROLE_STYLES[ROLE_VISITOR])
        prefix = style["prefix"]
        if self.face_name:
            return f"{prefix}: {self.face_name} ({self.face_id})"
        return f"{prefix} #{self.track_id}"

    @property
    def box_color(self) -> tuple:
        return ROLE_STYLES.get(self.role, ROLE_STYLES[ROLE_VISITOR])["color"]

    @property
    def is_worker(self) -> bool:
        return self.role in (ROLE_KNOWN_WORKER, ROLE_UNKNOWN_WORKER)

    @property
    def is_known(self) -> bool:
        return self.role in (ROLE_KNOWN_WORKER, ROLE_KNOWN_VISITOR)


class PersonClassifier:
    """
    Fuses face and uniform signals into a final role.

    Uses temporal smoothing: a person isn't flipped worker↔visitor on a
    single frame—the uniform signal must persist for `confirm_frames`
    consecutive frames before the role changes.
    """

    def __init__(self, confirm_frames: int = 5):
        """
        Parameters
        ----------
        confirm_frames : number of consecutive frames a uniform signal
                         must hold before we commit to the role change.
        """
        self.confirm_frames = confirm_frames
        self._people: dict[int, PersonInfo] = {}   # track_id → PersonInfo

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, track_id: int,
               face_name: Optional[str] = None,
               face_id: Optional[str] = None,
               face_confidence: float = 0.0,
               uniform_result: Optional[dict] = None) -> PersonInfo:
        """
        Update the classification for a tracked person.

        Parameters
        ----------
        track_id          : YOLO/ByteTrack track ID
        face_name         : recognised name (or None)
        face_id           : person ID string (or None)
        face_confidence   : face match confidence 0–100
        uniform_result    : dict from UniformDetector.classify()

        Returns
        -------
        PersonInfo with the up-to-date role.
        """
        if track_id not in self._people:
            self._people[track_id] = PersonInfo(track_id=track_id)
        person = self._people[track_id]

        # ── Update face info (sticky — once identified, stays) ─────────
        if face_name and face_confidence > person.face_confidence:
            person.face_name       = face_name
            person.face_id         = face_id
            person.face_confidence = face_confidence

        # ── Update uniform info with temporal smoothing ────────────────
        if uniform_result:
            is_match = uniform_result.get("role") == "worker"
            person.uniform_ratio      = uniform_result.get("match_ratio", 0.0)
            person.uniform_label      = uniform_result.get("profile_label", "")
            person.uniform_confidence = uniform_result.get("confidence", 0.0)

            if is_match:
                person._uniform_streak += 1
                person._visitor_streak  = 0
            else:
                person._visitor_streak += 1
                person._uniform_streak  = 0

            # Commit only after streak exceeds threshold
            if person._uniform_streak >= self.confirm_frames:
                person.uniform_match = True
            elif person._visitor_streak >= self.confirm_frames:
                person.uniform_match = False

        # ── Compute final role ─────────────────────────────────────────
        has_face    = person.face_name is not None
        has_uniform = person.uniform_match

        if has_face and has_uniform:
            person.role = ROLE_KNOWN_WORKER
        elif not has_face and has_uniform:
            person.role = ROLE_UNKNOWN_WORKER
        elif has_face and not has_uniform:
            person.role = ROLE_KNOWN_VISITOR
        else:
            person.role = ROLE_VISITOR

        return person

    def get(self, track_id: int) -> Optional[PersonInfo]:
        """Retrieve current info for a track, or None."""
        return self._people.get(track_id)

    def remove(self, track_id: int):
        """Clean up when a track disappears."""
        self._people.pop(track_id, None)

    def active_counts(self) -> dict:
        """Return counts by role among currently tracked people."""
        counts = {ROLE_KNOWN_WORKER: 0, ROLE_UNKNOWN_WORKER: 0,
                  ROLE_KNOWN_VISITOR: 0, ROLE_VISITOR: 0}
        for p in self._people.values():
            counts[p.role] = counts.get(p.role, 0) + 1
        return counts

    @property
    def worker_count(self) -> int:
        return sum(1 for p in self._people.values() if p.is_worker)

    @property
    def visitor_count(self) -> int:
        return sum(1 for p in self._people.values() if not p.is_worker)
