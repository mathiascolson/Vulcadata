# scripts/upload_existing_extractions_to_s3.py

import os
from pathlib import Path
from dotenv import load_dotenv
import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

S3_BUCKET = "vulcadata"

s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))

local_base = PROJECT_ROOT / "data" / "extraction"

files_to_upload = list((local_base / "processed_csv").glob("*.csv"))
files_to_upload += list((local_base / "quality_reports").glob("*.json"))

for path in files_to_upload:
    name = path.name

    if path.suffix == ".csv":
        eruption_id = name.replace("_filtered_1_16Hz_aggregated_1min_with_fi.csv", "")
        s3_key = f"volcano/processed/aggregated_csv/{eruption_id}/{name}"
    elif path.suffix == ".json":
        eruption_id = name.replace("_mseed_validation.json", "").replace("_csv_validation.json", "")
        s3_key = f"volcano/quality/extraction_reports/{eruption_id}/{name}"
    else:
        continue

    print(f"Upload : {path} -> s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(str(path), S3_BUCKET, s3_key)

print("Uploads terminés.")