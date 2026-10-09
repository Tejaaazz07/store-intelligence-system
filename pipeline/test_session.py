# PROMPT:
# Generate a simple validation test for KPI calculations in the Store Intelligence System.

# CHANGES MADE:
# Modified imports and adjusted assertions to match project structure.

from session_tracker import process_visitor, visitor_exit

print(process_visitor("1"))
print(visitor_exit("1"))
print(process_visitor("1"))
@app.get("/funnel")
def funnel():

    from database import SessionLocal
    from models import Event

    db = SessionLocal()

    entries = db.query(Event).filter(
        Event.event_type == "entry"
    ).count()

    exits = db.query(Event).filter(
        Event.event_type == "exit"
    ).count()

    active = entries - exits

    db.close()

    return {
        "entered": entries,
        "active": active,
        "exited": exits
    }