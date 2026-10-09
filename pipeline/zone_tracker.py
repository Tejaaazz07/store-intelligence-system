from zones import get_zone

person_zones = {}

def track_zone(person_id, x_center):

    current_zone = get_zone(x_center)

    if person_id not in person_zones:
        person_zones[person_id] = current_zone

        return {
            "event_type": "ZONE_ENTER",
            "person_id": str(person_id),
            "zone": current_zone
        }

    previous_zone = person_zones[person_id]

    if previous_zone != current_zone:

        person_zones[person_id] = current_zone

        return {
            "event_type": "ZONE_ENTER",
            "person_id": str(person_id),
            "zone": current_zone
        }

    return None