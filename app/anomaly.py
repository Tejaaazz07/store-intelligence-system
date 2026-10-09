import sqlite3

DB_NAME = "store.db"


def detect_anomaly():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM events
    """)

    total_events = cursor.fetchone()[0]

    conn.close()

    if total_events > 10:
        return {
            "anomaly": True,
            "reason": "High event volume detected"
        }

    return {
        "anomaly": False,
        "reason": "Normal activity"
    }


if __name__ == "__main__":
    print(detect_anomaly())