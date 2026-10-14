"""
person_classifier.py
====================
Role & Category classifier based on face identification:
  • Lakshana  → ★ Hacker
  • Shirley   → ★ Hustler
  • Mehaa     → ★ Hacker
  • Other registered staff → ★ Worker
  • Unregistered face / unknown → Customer
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict

CONFIG_PATH = os.path.join("models", "workers_config.json")

# Default categories mapping (name lowercase -> category)
DEFAULT_WORKER_ROLES: Dict[str, str] = {
    "lakshana": "Hacker",
    "shirley":  "Hustler",
    "mehaa":    "Hacker",
}

# Visual styling for each role/category
# BGR colour, display prefix, emoji badge
ROLE_STYLES = {
    "Hacker":   {"color": (0, 255, 128),  "prefix": "★ Hacker",  "emoji": "💻", "badge": "H"},
    "Hustler":  {"color": (255, 105, 180), "prefix": "★ Hustler", "emoji": "⚡", "badge": "H"},
    "Worker":   {"color": (0, 215, 255),  "prefix": "★ Worker",  "emoji": "👷", "badge": "W"},
    "Customer": {"color": (255, 140, 0),    "prefix": "Customer",   "emoji": "🛍️", "badge": "C"},
}


def load_worker_categories() -> Dict[str, str]:
    """Load worker categories from JSON configuration file."""
    categories = dict(DEFAULT_WORKER_ROLES)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, meta in data.items():
                if isinstance(meta, dict) and "category" in meta:
                    categories[name.lower()] = meta["category"]
                elif isinstance(meta, str):
                    categories[name.lower()] = meta
        except Exception:
            pass
    return categories


@dataclass
class PersonInfo:
    """Everything we know about a tracked person in the current session."""
    track_id:        int
    role:            str = "customer"        # "worker" | "customer"
    category:        str = "Customer"        # "Hacker" | "Hustler" | "Worker" | "Customer"
    face_name:       Optional[str] = None
    face_id:         Optional[str] = None
    face_confidence: float = 0.0

    # ── Display helpers ────────────────────────────────────────────────────

    @property
    def style(self) -> dict:
        return ROLE_STYLES.get(self.category, ROLE_STYLES["Worker" if self.is_worker else "Customer"])

    @property
    def display_label(self) -> str:
        if self.face_name:
            prefix = self.style["prefix"]
            return f"{prefix}: {self.face_name} ({self.face_id})"
        return f"Customer #{self.track_id}"

    @property
    def box_color(self) -> tuple:
        return self.style["color"]

    @property
    def badge_text(self) -> str:
        return self.style.get("badge", "W" if self.is_worker else "C")

    @property
    def emoji(self) -> str:
        return self.style.get("emoji", "👷" if self.is_worker else "🛍️")

    @property
    def is_worker(self) -> bool:
        return self.role == "worker"

    @property
    def is_customer(self) -> bool:
        return self.role == "customer"


class PersonClassifier:
    """
    Classifies tracked individuals into Worker categories (Hacker, Hustler, etc.)
    or Customer.
    """

    def __init__(self):
        self._people: dict[int, PersonInfo] = {}   # track_id → PersonInfo
        self._categories = load_worker_categories()

    def reload_categories(self):
        self._categories = load_worker_categories()

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, track_id: int,
               face_name: Optional[str] = None,
               face_id: Optional[str] = None,
               face_confidence: float = 0.0,
               **kwargs) -> PersonInfo:
        """
        Update the classification for a tracked person.
        """
        if track_id not in self._people:
            self._people[track_id] = PersonInfo(track_id=track_id)
        person = self._people[track_id]

        if face_name and face_confidence > person.face_confidence:
            person.face_name       = face_name
            person.face_id         = face_id
            person.face_confidence = face_confidence
            person.role            = "worker"

            # Determine specific category (e.g. Hacker, Hustler)
            cat = self._categories.get(face_name.lower(), "Worker")
            person.category = cat
        elif person.face_name is None:
            person.role     = "customer"
            person.category = "Customer"

        return person

    def get(self, track_id: int) -> Optional[PersonInfo]:
        """Retrieve current info for a track, or None."""
        return self._people.get(track_id)

    def remove(self, track_id: int):
        """Clean up when a track disappears."""
        self._people.pop(track_id, None)

    def active_counts(self) -> dict:
        """Return counts by role/category among currently tracked people."""
        counts = {"workers": 0, "customers": 0, "hackers": 0, "hustlers": 0}
        for p in self._people.values():
            if p.is_worker:
                counts["workers"] += 1
                if p.category == "Hacker":
                    counts["hackers"] += 1
                elif p.category == "Hustler":
                    counts["hustlers"] += 1
            else:
                counts["customers"] += 1
        return counts

    @property
    def worker_count(self) -> int:
        return sum(1 for p in self._people.values() if p.is_worker)

    @property
    def customer_count(self) -> int:
        return sum(1 for p in self._people.values() if p.is_customer)
