from __future__ import annotations

SHORT_NAMES: dict[str, str] = {
    "H3O^1": "H3O+",
    "°OH^0": "OH",
    "OH^-1": "OH-",
    "e_aq^-1": "e_aq",
    "H^0": "H",
    "H_2^0": "H2",
    "H2O2^0": "H2O2",
    "HO_2^-1": "HO2-",
    "O_2^0": "O2",
    "°O^0": "O",
    "O_2^-1": "O2-",
    "HO_2°^0": "HO2",
}


def short_name(raw: str) -> str:
    """Return the short species label, or ``raw`` unchanged if unknown."""
    return SHORT_NAMES.get(raw, raw)
