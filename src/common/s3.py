from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class S3ClientProtocol(Protocol):
    """
    Minimal protocol for an S3-compatible client.

    This keeps the module testable without real AWS credentials or network calls.
    """

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        ...

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        ...


@dataclass(frozen=True)
class S3Location:
    """
    Validated S3 object location.
    """

    bucket: str
    key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket", validate_bucket_name(self.bucket))
        object.__setattr__(self, "key", normalize_s3_key(self.key))

    @property
    def uri(self) -> str:
        return build_s3_uri(bucket=self.bucket, key=self.key)


def validate_bucket_name(bucket: str) -> str:
    """
    Validate a bucket name used by the project.

    This function intentionally performs lightweight validation only.
    Full AWS bucket-name validation is not needed here because infrastructure
    naming belongs to configuration and deployment, not business code.
    """
    if not isinstance(bucket, str):
        raise TypeError("bucket must be a string.")

    cleaned_bucket = bucket.strip()

    if not cleaned_bucket:
        raise ValueError("bucket cannot be empty.")

    if cleaned_bucket.startswith("s3://"):
        raise ValueError("bucket must be a bucket name, not an S3 URI.")

    if "/" in cleaned_bucket:
        raise ValueError("bucket must not contain '/'.")

    return cleaned_bucket


def normalize_s3_key(key: str) -> str:
    """
    Normalize an S3 object key.

    The project uses S3 keys as logical relative paths:
    - no leading slash;
    - forward slashes only;
    - no empty path segments;
    - no '.' or '..' path traversal segments.
    """
    if not isinstance(key, str):
        raise TypeError("key must be a string.")

    cleaned_key = key.strip().replace("\\", "/")

    if cleaned_key.startswith("s3://"):
        raise ValueError("key must be an object key, not an S3 URI.")

    cleaned_key = cleaned_key.strip("/")

    if not cleaned_key:
        raise ValueError("key cannot be empty.")

    parts = [part for part in cleaned_key.split("/") if part]

    if any(part in {".", ".."} for part in parts):
        raise ValueError("key must not contain '.' or '..' path segments.")

    return "/".join(parts)


def normalize_s3_prefix(prefix: str | None) -> str:
    """
    Normalize an S3 prefix.

    Unlike object keys, an empty prefix is valid when listing a bucket.
    """
    if prefix is None:
        return ""

    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string or None.")

    cleaned_prefix = prefix.strip().replace("\\", "/").strip("/")

    if not cleaned_prefix:
        return ""

    normalized_prefix = normalize_s3_key(cleaned_prefix)

    if prefix.replace("\\", "/").endswith("/"):
        return f"{normalized_prefix}/"

    return normalized_prefix


def build_s3_uri(bucket: str, key: str) -> str:
    """
    Build a canonical S3 URI.

    Example:
    s3://vulcadata/predictions/latest/prediction.json
    """
    validated_bucket = validate_bucket_name(bucket)
    normalized_key = normalize_s3_key(key)

    return f"s3://{validated_bucket}/{normalized_key}"


def parse_s3_uri(uri: str) -> S3Location:
    """
    Parse an S3 URI into bucket and key.

    Accepted format:
    s3://bucket/path/to/object.json
    """
    if not isinstance(uri, str):
        raise TypeError("uri must be a string.")

    cleaned_uri = uri.strip()

    if not cleaned_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'.")

    without_scheme = cleaned_uri.removeprefix("s3://")

    if "/" not in without_scheme:
        raise ValueError("S3 URI must include both bucket and key.")

    bucket, key = without_scheme.split("/", maxsplit=1)

    return S3Location(bucket=bucket, key=key)


def create_s3_client(region_name: str | None = None, endpoint_url: str | None = None) -> Any:
    """
    Create a boto3 S3 client.

    boto3 is imported lazily so local unit tests can run without requiring
    AWS dependencies or credentials.
    """
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required to create a real S3 client. "
            "Install it or inject a test client."
        ) from exc

    kwargs: dict[str, str] = {}

    if region_name:
        kwargs["region_name"] = region_name

    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **kwargs)


def s3_object_exists(client: S3ClientProtocol, bucket: str, key: str) -> bool:
    """
    Return True if an S3 object exists.

    Missing objects return False.
    Other client errors are re-raised.
    """
    validated_bucket = validate_bucket_name(bucket)
    normalized_key = normalize_s3_key(key)

    try:
        client.head_object(Bucket=validated_bucket, Key=normalized_key)
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise

    return True


def upload_text(
    client: S3ClientProtocol,
    bucket: str,
    key: str,
    text: str,
    *,
    encoding: str = "utf-8",
    content_type: str = "text/plain",
) -> S3Location:
    """
    Upload text content to S3.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    location = S3Location(bucket=bucket, key=key)

    client.put_object(
        Bucket=location.bucket,
        Key=location.key,
        Body=text.encode(encoding),
        ContentType=content_type,
    )

    return location


def download_text(
    client: S3ClientProtocol,
    bucket: str,
    key: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """
    Download text content from S3.
    """
    location = S3Location(bucket=bucket, key=key)

    response = client.get_object(Bucket=location.bucket, Key=location.key)
    body = response["Body"].read()

    if isinstance(body, bytes):
        return body.decode(encoding)

    if isinstance(body, str):
        return body

    raise TypeError("S3 object body must be bytes or string.")


def upload_json(
    client: S3ClientProtocol,
    bucket: str,
    key: str,
    data: Any,
    *,
    indent: int = 2,
) -> S3Location:
    """
    Serialize JSON-compatible data and upload it to S3.
    """
    json_text = json.dumps(
        data,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )

    return upload_text(
        client=client,
        bucket=bucket,
        key=key,
        text=json_text,
        content_type="application/json",
    )


def download_json(client: S3ClientProtocol, bucket: str, key: str) -> Any:
    """
    Download and deserialize JSON content from S3.
    """
    text = download_text(client=client, bucket=bucket, key=key)
    return json.loads(text)


def upload_file(
    client: S3ClientProtocol,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> S3Location:
    """
    Upload a local file to S3.
    """
    path = Path(local_path)

    if not path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {path}")

    location = S3Location(bucket=bucket, key=key)

    client.upload_file(
        Filename=str(path),
        Bucket=location.bucket,
        Key=location.key,
    )

    return location


def download_file(
    client: S3ClientProtocol,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> Path:
    """
    Download an S3 object to a local file.
    """
    location = S3Location(bucket=bucket, key=key)
    path = Path(local_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    client.download_file(
        Bucket=location.bucket,
        Key=location.key,
        Filename=str(path),
    )

    return path


def list_s3_keys(
    client: S3ClientProtocol,
    bucket: str,
    prefix: str | None = None,
    *,
    max_keys: int | None = None,
) -> list[str]:
    """
    List object keys under an S3 prefix.

    Pagination is handled through list_objects_v2.
    """
    validated_bucket = validate_bucket_name(bucket)
    normalized_prefix = normalize_s3_prefix(prefix)

    if max_keys is not None and max_keys <= 0:
        raise ValueError("max_keys must be strictly positive when provided.")

    request: dict[str, Any] = {
        "Bucket": validated_bucket,
        "Prefix": normalized_prefix,
    }

    if max_keys is not None:
        request["MaxKeys"] = min(max_keys, 1000)

    keys: list[str] = []

    while True:
        response = client.list_objects_v2(**request)

        for item in response.get("Contents", []):
            keys.append(item["Key"])

            if max_keys is not None and len(keys) >= max_keys:
                return keys

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

        if not continuation_token:
            break

        request["ContinuationToken"] = continuation_token

    return keys


def _is_not_found_error(exc: Exception) -> bool:
    """
    Detect common S3 missing-object errors without depending directly on botocore.
    """
    response = getattr(exc, "response", None)

    if isinstance(response, dict):
        error = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})

        error_code = str(error.get("Code", ""))
        status_code = metadata.get("HTTPStatusCode")

        if status_code == 404:
            return True

        if error_code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}:
            return True

    direct_code = str(getattr(exc, "code", ""))
    direct_status_code = getattr(exc, "status_code", None)

    if direct_status_code == 404:
        return True

    if direct_code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}:
        return True

    return False