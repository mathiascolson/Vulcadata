from types import SimpleNamespace

import pandas as pd
import pytest

import src.extraction.extract_volcano_periods as extraction


def test_extraction_main_reads_semicolon_periods_csv(tmp_path, monkeypatch):
    periods_path = tmp_path / "extraction_periods.csv"
    output_dir = tmp_path / "extraction_output"

    periods_path.write_text(
        "\n".join(
            [
                "period_id;period_type;period_start_utc;period_end_utc;network;stations;channels",
                "quiet_test;quiet;2020-01-01T00:00:00Z;2020-01-01T03:00:00Z;PF;CSS,DSO;HHZ,EHZ",
                "inference_test;inference;2020-01-02T00:00:00Z;2020-01-02T03:00:00Z;PF;CSS;HHZ",
            ]
        ),
        encoding="utf-8",
    )

    seen_periods = []

    def fake_process_period(row, args):
        seen_periods.append(row.to_dict())
        return {
            "period_id": row["period_id"],
            "period_type": row["period_type"],
            "status": "success",
        }

    monkeypatch.setattr(extraction, "process_period", fake_process_period)

    args = SimpleNamespace(
        periods=str(periods_path),
        output_dir=str(output_dir),
        base_url="https://example.invalid/fdsn",
        timeout=1,
        eps=1e-12,
        force_download=False,
    )

    extraction.main(args)

    assert [row["period_id"] for row in seen_periods] == ["quiet_test", "inference_test"]
    assert [row["period_type"] for row in seen_periods] == ["quiet", "inference"]
    assert (output_dir / "quality_reports" / "extraction_summary.json").exists()


def test_validate_periods_accepts_quiet_and_inference_without_eruption_start():
    periods = pd.DataFrame(
        [
            {
                "period_id": "quiet_test",
                "period_type": "quiet",
                "period_start_utc": "2020-01-01T00:00:00Z",
                "period_end_utc": "2020-01-01T03:00:00Z",
                "network": "PF",
                "stations": "CSS,DSO",
                "channels": "HHZ,EHZ",
            },
            {
                "period_id": "inference_test",
                "period_type": "inference",
                "period_start_utc": "2020-01-02T00:00:00Z",
                "period_end_utc": "2020-01-02T03:00:00Z",
                "network": "PF",
                "stations": "CSS",
                "channels": "HHZ",
            },
        ]
    )

    validated = extraction.validate_periods(periods)

    assert validated["period_type"].tolist() == ["quiet", "inference"]
    assert "eruption_start_utc" in validated.columns
    assert "eruption_end_utc" in validated.columns
    assert "split" in validated.columns


def test_validate_periods_rejects_missing_required_columns():
    periods = pd.DataFrame(
        [
            {
                "period_id": "bad_period",
                "period_type": "quiet",
            }
        ]
    )

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        extraction.validate_periods(periods)


def test_validate_periods_rejects_duplicate_period_id():
    periods = pd.DataFrame(
        [
            {
                "period_id": "duplicated_period",
                "period_type": "quiet",
                "period_start_utc": "2020-01-01T00:00:00Z",
                "period_end_utc": "2020-01-01T03:00:00Z",
                "network": "PF",
                "stations": "CSS",
                "channels": "HHZ",
            },
            {
                "period_id": "duplicated_period",
                "period_type": "quiet",
                "period_start_utc": "2020-01-02T00:00:00Z",
                "period_end_utc": "2020-01-02T03:00:00Z",
                "network": "PF",
                "stations": "CSS",
                "channels": "HHZ",
            },
        ]
    )

    with pytest.raises(ValueError, match="period_id dupliqué"):
        extraction.validate_periods(periods)
