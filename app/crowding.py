import sqlite3

DB_NAME = "store.db"


def check_crowding(threshold=3):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT person_id)
        FROM events
        WHERE event_type='entry'
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return {
        "current_people": count,
        "crowding_alert": count > threshold
    }


if __name__ == "__main__":
    print(check_crowding())