def is_locked(value):
    """Interpret database boolean values without treating the string 'false' as true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "locked"}


def unlocked_records(records):
    """Return a copy of dataset rows that are not temporarily locked."""
    if "is_locked" not in records.columns:
        return records.copy()
    mask = ~records["is_locked"].map(is_locked)
    return records.loc[mask].copy()