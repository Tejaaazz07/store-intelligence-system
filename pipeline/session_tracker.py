import uuid

active_visitors = set()
seen_visitors = set()

def process_visitor(visitor_id):
    events = []

    if visitor_id not in active_visitors:

        if visitor_id in seen_visitors:
            events.append({
                "event_type": "REENTRY",
                "visitor_id": visitor_id
            })
        else:
            events.append({
                "event_type": "ENTRY",
                "visitor_id": visitor_id
            })

        active_visitors.add(visitor_id)
        seen_visitors.add(visitor_id)

    return events


def visitor_exit(visitor_id):

    if visitor_id in active_visitors:
        active_visitors.remove(visitor_id)

        return {
            "event_type": "EXIT",
            "visitor_id": visitor_id
        }

    return None