import sqlite3

DB_NAME = "store.db"


def save_event(event_type, person_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events (event_type, person_id, zone, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    """, (event_type, str(person_id), "store"))

    conn.commit()
    conn.close()

    print(f"Saved {event_type} event for Person {person_id}")


tracked_people = [1, 5]

for person_id in tracked_people:
    save_event("entry", person_id)

print("Event generation complete")