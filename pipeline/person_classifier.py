"""
person_classifier.py
====================
Role classifier based on face identification:
  • Face in database (known staff)  → WORKER
  • Face not in database / unknown   → CUSTOMER
"""

from dataclasses import dataclass
from typing import Optional


# ── Role constants ─────────────────────────────────────────────────────────────
ROLE_WORKER   = "worker"
ROLE_CUSTOMER = "customer"

# Visual styling for each role (BGR colour, display label prefix, emoji)
ROLE_STYLES = {
    ROLE_WORKER:   {"color": (0, 215, 255), "prefix": "★ Worker", "emoji": "👷"},
    ROLE_CUSTOMER: {"color": (255, 140, 0),  "prefix": "Customer", "emoji": "🛍️"},
}


@dataclass
class PersonInfo:
    """Everything we know about a tracked person in the current session."""
    track_id:        int
    role:            str = ROLE_CUSTOMER
    # Face recognition
    face_name:       Optional[str] = None
    face_id:         Optional[str] = None
    face_confidence: float = 0.0

    # ── Display helpers ────────────────────────────────────────────────────

    @property
    def display_label(self) -> str:
        if self.face_name:
            return f"★ Worker: {self.face_name} ({self.face_id})"
        return f"Customer #{self.track_id}"

    @property
    def box_color(self) -> tuple:
        return ROLE_STYLES.get(self.role, ROLE_STYLES[ROLE_CUSTOMER])["color"]

    @property
    def is_worker(self) -> bool:
        return self.role == ROLE_WORKER

    @property
    def is_customer(self) -> bool:
        return self.role == ROLE_CUSTOMER


class PersonClassifier:
    """
    Classifies tracked individuals:
      - Any person matching registered face records is a WORKER.
      - Everyone else is a CUSTOMER.
    """

    def __init__(self):
        self._people: dict[int, PersonInfo] = {}   # track_id → PersonInfo

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, track_id: int,
               face_name: Optional[str] = None,
               face_id: Optional[str] = None,
               face_confidence: float = 0.0,
               **kwargs) -> PersonInfo:
        """
        Update the classification for a tracked person.

        Parameters
        ----------
        track_id        : YOLO/ByteTrack track ID
        face_name       : recognised name (or None)
        face_id         : person/employee ID string (or None)
        face_confidence : face match confidence 0–100

        Returns
        -------
        PersonInfo with the role (worker or customer).
        """
        if track_id not in self._people:
            self._people[track_id] = PersonInfo(track_id=track_id)
        person = self._people[track_id]

        # Update face info (sticky — once identified as a worker, stays worker)
        if face_name and face_confidence > person.face_confidence:
            person.face_name       = face_name
            person.face_id         = face_id
            person.face_confidence = face_confidence
            person.role            = ROLE_WORKER
        elif person.face_name is None:
            person.role = ROLE_CUSTOMER

        return person

    def get(self, track_id: int) -> Optional[PersonInfo]:
        """Retrieve current info for a track, or None."""
        return self._people.get(track_id)

    def remove(self, track_id: int):
        """Clean up when a track disappears."""
        self._people.pop(track_id, None)

    def active_counts(self) -> dict:
        """Return counts by role among currently tracked people."""
        counts = {ROLE_WORKER: 0, ROLE_CUSTOMER: 0}
        for p in self._people.values():
            counts[p.role] = counts.get(p.role, 0) + 1
        return counts

    @property
    def worker_count(self) -> int:
        return sum(1 for p in self._people.values() if p.is_worker)

    @property
    def customer_count(self) -> int:
        return sum(1 for p in self._people.values() if p.is_customer)
