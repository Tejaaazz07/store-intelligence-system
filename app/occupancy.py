# PROMPT:
# Generate a simple validation test for KPI calculations in the Store Intelligence System.

# CHANGES MADE:
# Modified imports and adjusted assertions to match project structure.

peak_people = 0

def update_peak(current_people):

    global peak_people

    if current_people > peak_people:
        peak_people = current_people

    return peak_people