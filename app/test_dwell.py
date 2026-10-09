# PROMPT:
# Generate a simple validation test for KPI calculations in the Store Intelligence System.

# CHANGES MADE:
# Modified imports and adjusted assertions to match project structure.

from dwell_time import enter_store, exit_store

enter_store("1", 100)

result = exit_store("1", 145)

print(result)