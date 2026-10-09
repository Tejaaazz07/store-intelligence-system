# PROMPT:
# Generate a simple validation test for KPI calculations in the Store Intelligence System.

# CHANGES MADE:
# Modified imports and adjusted assertions to match project structure.

from kpi import generate_kpis

def test_kpi_generation():
    result = generate_kpis()
    assert result is not None