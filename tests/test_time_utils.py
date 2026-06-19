from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.common.time_utils import (
    build_utc_window,
    ensure_utc,
    floor_datetime_to_minutes,
    format_compact_utc,
    format_utc_datetime,
    is_timezone_aware,
    parse_utc_datetime,
    utc_now,
)


def test_utc_now_returns_timezone_aware_utc_datetime_without_microseconds() -> None:
    value = utc_now()

    assert value.tzinfo is UTC
    assert value.utcoffset() == timedelta(0)
    assert value.microsecond == 0


def test_is_timezone_aware_detects_aware_and_naive_datetimes() -> None:
    aware = datetime(2026, 6, 17, 8, 30, tzinfo=UTC)
    naive = datetime(2026, 6, 17, 8, 30)

    assert is_timezone_aware(aware) is True
    assert is_timezone_aware(naive) is False


def test_ensure_utc_converts_timezone_aware_datetime_to_utc() -> None:
    paris_tz = timezone(timedelta(hours=2))
    value = datetime(2026, 6, 17, 8, 30, tzinfo=paris_tz)

    result = ensure_utc(value)

    assert result == datetime(2026, 6, 17, 6, 30, tzinfo=UTC)


def test_ensure_utc_rejects_naive_datetime_by_default() -> None:
    value = datetime(2026, 6, 17, 8, 30)

    with pytest.raises(ValueError, match="Naive datetimes are not allowed"):
        ensure_utc(value)


def test_ensure_utc_can_interpret_naive_datetime_as_utc_when_explicitly_allowed() -> None:
    value = datetime(2026, 6, 17, 8, 30)

    result = ensure_utc(value, allow_naive=True)

    assert result == datetime(2026, 6, 17, 8, 30, tzinfo=UTC)


def test_parse_utc_datetime_accepts_z_suffix() -> None:
    result = parse_utc_datetime("2026-06-17T06:30:00Z")

    assert result == datetime(2026, 6, 17, 6, 30, tzinfo=UTC)


def test_parse_utc_datetime_converts_offset_to_utc() -> None:
    result = parse_utc_datetime("2026-06-17T08:30:00+02:00")

    assert result == datetime(2026, 6, 17, 6, 30, tzinfo=UTC)


def test_parse_utc_datetime_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Datetime string cannot be empty"):
        parse_utc_datetime("   ")


def test_parse_utc_datetime_rejects_naive_string_by_default() -> None:
    with pytest.raises(ValueError, match="Naive datetimes are not allowed"):
        parse_utc_datetime("2026-06-17T06:30:00")


def test_parse_utc_datetime_rejects_invalid_string() -> None:
    with pytest.raises(ValueError, match="Invalid ISO datetime"):
        parse_utc_datetime("not-a-date")


def test_format_utc_datetime_returns_z_suffix_without_microseconds() -> None:
    value = "2026-06-17T08:30:00.123456+02:00"

    result = format_utc_datetime(value)

    assert result == "2026-06-17T06:30:00Z"


def test_format_compact_utc_returns_filename_safe_timestamp() -> None:
    value = "2026-06-17T08:30:00+02:00"

    result = format_compact_utc(value)

    assert result == "20260617T063000Z"


def test_build_utc_window_returns_start_and_end_in_utc() -> None:
    start, end = build_utc_window("2026-06-17T12:00:00Z", window_hours=24)

    assert start == datetime(2026, 6, 16, 12, 0, tzinfo=UTC)
    assert end == datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


def test_build_utc_window_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        build_utc_window("2026-06-17T12:00:00Z", window_hours=0)


def test_floor_datetime_to_minutes_floors_to_previous_bucket() -> None:
    result = floor_datetime_to_minutes("2026-06-17T06:37:45Z", step_minutes=10)

    assert result == datetime(2026, 6, 17, 6, 30, tzinfo=UTC)


def test_floor_datetime_to_minutes_converts_to_utc_before_flooring() -> None:
    result = floor_datetime_to_minutes("2026-06-17T08:37:45+02:00", step_minutes=10)

    assert result == datetime(2026, 6, 17, 6, 30, tzinfo=UTC)


def test_floor_datetime_to_minutes_rejects_invalid_step() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        floor_datetime_to_minutes("2026-06-17T06:37:45Z", step_minutes=0)