visitor_times = {}

def enter_store(visitor_id, timestamp):

    visitor_times[visitor_id] = {
        "entry": timestamp
    }

def exit_store(visitor_id, timestamp):

    if visitor_id in visitor_times:

        entry_time = visitor_times[visitor_id]["entry"]

        dwell_time = timestamp - entry_time

        return {
            "visitor_id": visitor_id,
            "dwell_time_seconds": dwell_time
        }

    return None