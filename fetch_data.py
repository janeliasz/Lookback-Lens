from google.cloud import storage
import os

client = storage.Client.from_service_account_json(
    os.path.join("secrets", "storage-gcp-key.json")
)

bucket_name = "hallucination-detection"
bucket = client.bucket(bucket_name)

blob_f = bucket.blob("data/filtered_sample_1500.parquet").download_to_filename("data/hallu-ds.parquet")
