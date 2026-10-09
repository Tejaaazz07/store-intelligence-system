import sqlite3

DB_NAME = "store.db"


def get_total_customers():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(DISTINCT person_id)
        FROM events
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


if __name__ == "__main__":
    print("Total Customers:", get_total_customers())