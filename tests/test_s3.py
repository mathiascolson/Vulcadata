from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.common.s3 import (
    S3Location,
    build_s3_uri,
    download_file,
    download_json,
    download_text,
    list_s3_keys,
    normalize_s3_key,
    normalize_s3_prefix,
    parse_s3_uri,
    s3_object_exists,
    upload_file,
    upload_json,
    upload_text,
    validate_bucket_name,
)


class FakeS3Body:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content


class FakeS3Error(Exception):
    def __init__(self, code: str, status_code: int) -> None:
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }
        super().__init__(code)


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.content_types: dict[tuple[str, str], str | None] = {}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]

        if (bucket, key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        return {"ContentLength": len(self.objects[(bucket, key)])}

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        body = kwargs["Body"]
        content_type = kwargs.get("ContentType")

        if isinstance(body, str):
            body = body.encode("utf-8")

        self.objects[(bucket, key)] = body
        self.content_types[(bucket, key)] = content_type

        return {"ETag": "fake-etag"}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]

        if (bucket, key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        return {"Body": FakeS3Body(self.objects[(bucket, key)])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket = kwargs["Bucket"]
        prefix = kwargs.get("Prefix", "")
        max_keys = kwargs.get("MaxKeys")
        continuation_token = kwargs.get("ContinuationToken")

        matching_keys = sorted(
            key
            for current_bucket, key in self.objects
            if current_bucket == bucket and key.startswith(prefix)
        )

        start_index = int(continuation_token) if continuation_token else 0
        end_index = len(matching_keys)

        if max_keys is not None:
            end_index = min(start_index + max_keys, len(matching_keys))

        selected_keys = matching_keys[start_index:end_index]
        is_truncated = end_index < len(matching_keys)

        response: dict[str, Any] = {
            "Contents": [{"Key": key} for key in selected_keys],
            "IsTruncated": is_truncated,
        }

        if is_truncated:
            response["NextContinuationToken"] = str(end_index)

        return response

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        if (Bucket, Key) not in self.objects:
            raise FakeS3Error(code="NoSuchKey", status_code=404)

        Path(Filename).write_bytes(self.objects[(Bucket, Key)])


def test_validate_bucket_name_accepts_valid_bucket() -> None:
    assert validate_bucket_name("vulcadata") == "vulcadata"


def test_validate_bucket_name_rejects_empty_bucket() -> None:
    with pytest.raises(ValueError, match="bucket cannot be empty"):
        validate_bucket_name("   ")


def test_validate_bucket_name_rejects_s3_uri() -> None:
    with pytest.raises(ValueError, match="not an S3 URI"):
        validate_bucket_name("s3://vulcadata")


def test_normalize_s3_key_normalizes_path_separators_and_slashes() -> None:
    result = normalize_s3_key("/predictions\\history//run/prediction.json")

    assert result == "predictions/history/run/prediction.json"


def test_normalize_s3_key_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key cannot be empty"):
        normalize_s3_key(" / ")


def test_normalize_s3_key_rejects_s3_uri() -> None:
    with pytest.raises(ValueError, match="not an S3 URI"):
        normalize_s3_key("s3://vulcadata/prediction.json")


def test_normalize_s3_key_rejects_path_traversal_segments() -> None:
    with pytest.raises(ValueError, match="path segments"):
        normalize_s3_key("predictions/../secret.json")


def test_normalize_s3_prefix_accepts_empty_prefix() -> None:
    assert normalize_s3_prefix(None) == ""
    assert normalize_s3_prefix("") == ""


def test_normalize_s3_prefix_preserves_trailing_slash() -> None:
    result = normalize_s3_prefix("/reports/inference/")

    assert result == "reports/inference/"


def test_build_s3_uri_returns_canonical_uri() -> None:
    result = build_s3_uri(
        bucket="vulcadata",
        key="/predictions/latest/prediction.json",
    )

    assert result == "s3://vulcadata/predictions/latest/prediction.json"


def test_parse_s3_uri_returns_location() -> None:
    location = parse_s3_uri("s3://vulcadata/predictions/latest/prediction.json")

    assert location == S3Location(
        bucket="vulcadata",
        key="predictions/latest/prediction.json",
    )
    assert location.uri == "s3://vulcadata/predictions/latest/prediction.json"


def test_parse_s3_uri_rejects_non_s3_uri() -> None:
    with pytest.raises(ValueError, match="must start with 's3://'"):
        parse_s3_uri("vulcadata/predictions/latest/prediction.json")


def test_upload_and_download_text_roundtrip() -> None:
    client = FakeS3Client()

    location = upload_text(
        client=client,
        bucket="vulcadata",
        key="predictions/latest/prediction.txt",
        text="alert=false",
    )

    result = download_text(
        client=client,
        bucket="vulcadata",
        key=location.key,
    )

    assert result == "alert=false"
    assert location.uri == "s3://vulcadata/predictions/latest/prediction.txt"


def test_upload_json_sets_json_content_type_and_downloads_data() -> None:
    client = FakeS3Client()
    data = {
        "eruption_id": "eruption_2019_10_25",
        "alert_24h": True,
        "p_alert_24h": 0.71,
    }

    location = upload_json(
        client=client,
        bucket="vulcadata",
        key="predictions/latest/prediction.json",
        data=data,
    )

    result = download_json(
        client=client,
        bucket="vulcadata",
        key=location.key,
    )

    assert result == data
    assert client.content_types[("vulcadata", location.key)] == "application/json"


def test_s3_object_exists_returns_true_for_existing_object() -> None:
    client = FakeS3Client()

    upload_text(
        client=client,
        bucket="vulcadata",
        key="reports/inference/report.txt",
        text="ok",
    )

    assert s3_object_exists(
        client=client,
        bucket="vulcadata",
        key="reports/inference/report.txt",
    ) is True


def test_s3_object_exists_returns_false_for_missing_object() -> None:
    client = FakeS3Client()

    assert s3_object_exists(
        client=client,
        bucket="vulcadata",
        key="missing/object.txt",
    ) is False


def test_list_s3_keys_filters_by_prefix() -> None:
    client = FakeS3Client()

    upload_text(client, "vulcadata", "predictions/latest/prediction.json", "latest")
    upload_text(client, "vulcadata", "predictions/history/run-1/prediction.json", "run-1")
    upload_text(client, "vulcadata", "reports/inference/report.json", "report")

    result = list_s3_keys(
        client=client,
        bucket="vulcadata",
        prefix="predictions/",
    )

    assert result == [
        "predictions/history/run-1/prediction.json",
        "predictions/latest/prediction.json",
    ]


def test_list_s3_keys_respects_max_keys() -> None:
    client = FakeS3Client()

    upload_text(client, "vulcadata", "reports/1.json", "1")
    upload_text(client, "vulcadata", "reports/2.json", "2")
    upload_text(client, "vulcadata", "reports/3.json", "3")

    result = list_s3_keys(
        client=client,
        bucket="vulcadata",
        prefix="reports/",
        max_keys=2,
    )

    assert result == [
        "reports/1.json",
        "reports/2.json",
    ]


def test_list_s3_keys_rejects_invalid_max_keys() -> None:
    client = FakeS3Client()

    with pytest.raises(ValueError, match="strictly positive"):
        list_s3_keys(
            client=client,
            bucket="vulcadata",
            prefix="reports/",
            max_keys=0,
        )


def test_upload_and_download_file_roundtrip(tmp_path: Path) -> None:
    client = FakeS3Client()

    source_file = tmp_path / "source.txt"
    destination_file = tmp_path / "nested" / "destination.txt"

    source_file.write_text("file-content", encoding="utf-8")

    location = upload_file(
        client=client,
        bucket="vulcadata",
        key="artifacts/source.txt",
        local_path=source_file,
    )

    downloaded_path = download_file(
        client=client,
        bucket="vulcadata",
        key=location.key,
        local_path=destination_file,
    )

    assert downloaded_path == destination_file
    assert destination_file.read_text(encoding="utf-8") == "file-content"


def test_upload_file_rejects_missing_local_file(tmp_path: Path) -> None:
    client = FakeS3Client()
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        upload_file(
            client=client,
            bucket="vulcadata",
            key="artifacts/missing.txt",
            local_path=missing_file,
        )