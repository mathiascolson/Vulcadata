from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Final


UTC_ISO_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"
UTC_COMPACT_FORMAT: Final[str] = "%Y%m%dT%H%M%SZ"


def utc_now() -> datetime:
    """
    Return the current UTC datetime.

    Microseconds are removed to keep timestamps stable for filenames,
    S3 keys, JSON outputs and logs.
    """
    return datetime.now(tz=UTC).replace(microsecond=0)


def is_timezone_aware(value: datetime) -> bool:
    """
    Return True if a datetime has usable timezone information.
    """
    return value.tzinfo is not None and value.utcoffset() is not None


def ensure_utc(value: datetime, *, allow_naive: bool = False) -> datetime:
    """
    Convert a timezone-aware datetime to UTC.

    By default, naive datetimes are rejected because they are ambiguous
    in an orchestrated batch pipeline.

    Parameters
    ----------
    value:
        Datetime to validate and convert.
    allow_naive:
        If True, naive datetimes are interpreted as UTC.

    Returns
    -------
    datetime
        Timezone-aware datetime normalized to UTC.
    """
    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime instance.")

    if not is_timezone_aware(value):
        if not allow_naive:
            raise ValueError("Naive datetimes are not allowed. Provide a timezone-aware datetime.")
        return value.replace(tzinfo=UTC)

    return value.astimezone(UTC)


def parse_utc_datetime(value: str | datetime, *, allow_naive: bool = False) -> datetime:
    """
    Parse an ISO datetime string or datetime object and normalize it to UTC.

    Accepted examples:
    - 2026-06-17T06:30:00Z
    - 2026-06-17T08:30:00+02:00
    - timezone-aware datetime objects

    Naive strings such as '2026-06-17T06:30:00' are rejected by default.
    """
    if isinstance(value, datetime):
        return ensure_utc(value, allow_naive=allow_naive)

    if not isinstance(value, str):
        raise TypeError("value must be a string or a datetime instance.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError("Datetime string cannot be empty.")

    # datetime.fromisoformat does not parse the trailing Z directly.
    if cleaned_value.endswith("Z"):
        cleaned_value = f"{cleaned_value[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(cleaned_value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime: {value!r}") from exc

    return ensure_utc(parsed, allow_naive=allow_naive)


def format_utc_datetime(value: str | datetime) -> str:
    """
    Format a datetime as an ISO UTC string with trailing Z.

    Output format:
    YYYY-MM-DDTHH:MM:SSZ
    """
    utc_value = parse_utc_datetime(value)
    return utc_value.replace(microsecond=0).strftime(UTC_ISO_FORMAT)


def format_compact_utc(value: str | datetime) -> str:
    """
    Format a datetime as a compact UTC timestamp suitable for filenames
    and S3 keys.

    Output format:
    YYYYMMDDTHHMMSSZ
    """
    utc_value = parse_utc_datetime(value)
    return utc_value.replace(microsecond=0).strftime(UTC_COMPACT_FORMAT)


def build_utc_window(end_time: str | datetime, window_hours: int | float) -> tuple[datetime, datetime]:
    """
    Build a UTC time window ending at `end_time`.

    Example:
    end_time = 2026-06-17T12:00:00Z
    window_hours = 24

    Returns:
    (
        2026-06-16T12:00:00Z,
        2026-06-17T12:00:00Z
    )
    """
    if not isinstance(window_hours, int | float):
        raise TypeError("window_hours must be an int or a float.")

    if not isfinite(window_hours) or window_hours <= 0:
        raise ValueError("window_hours must be a finite positive number.")

    end_utc = parse_utc_datetime(end_time)
    start_utc = end_utc - timedelta(hours=window_hours)

    return start_utc, end_utc


def floor_datetime_to_minutes(value: str | datetime, step_minutes: int) -> datetime:
    """
    Floor a datetime to the previous aligned minute bucket in UTC.

    Example:
    value = 2026-06-17T06:37:45Z
    step_minutes = 10

    Returns:
    2026-06-17T06:30:00Z
    """
    if not isinstance(step_minutes, int):
        raise TypeError("step_minutes must be an integer.")

    if step_minutes <= 0:
        raise ValueError("step_minutes must be strictly positive.")

    utc_value = parse_utc_datetime(value)

    minutes_since_midnight = utc_value.hour * 60 + utc_value.minute
    floored_minutes = (minutes_since_midnight // step_minutes) * step_minutes

    floored = utc_value.replace(hour=0, minute=0, second=0, microsecond=0)
    return floored + timedelta(minutes=floored_minutes)