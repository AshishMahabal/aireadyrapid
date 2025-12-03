import mlcroissant as mlc
import json
import hashlib
import os

BASE_DIR = "hackathon_dataset"
CSV_FILENAME = "master_index.csv"
LOCAL_CSV_PATH = os.path.join(BASE_DIR, CSV_FILENAME)

# calculating the SHA256 sum (required for FileObject)
def get_sha256(filepath):
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath} to calculate hash.")
        return ""

csv_sha256 = get_sha256(LOCAL_CSV_PATH)
print(f"Calculated SHA256 for {LOCAL_CSV_PATH}: {csv_sha256}")

ctx = mlc.Context()

distribution = [
    mlc.FileObject(
        id="master_index",
        name="master_index",
        content_url=CSV_FILENAME,
        encoding_formats=["text/csv"],
        sha256=csv_sha256
    ),
    mlc.FileSet(
        id="npy_images",
        name="npy_images",
        includes="images/*.npy", 
        encoding_formats=["application/x-numpy"]
    )
]

record_sets = [
    mlc.RecordSet(
        id="transient_candidates",
        name="transient_candidates",
        fields=[
            mlc.Field(
                id="transient_candidates/job_id",
                name="job_id",
                description="Job ID",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="job_id")
                )
            ),
            mlc.Field(
                id="transient_candidates/label",
                name="label",
                description="0 = Bogus, 1 = Real",
                data_types=[mlc.DataType.INTEGER],
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="label")
                )
            ),
            mlc.Field(
                id="transient_candidates/x",
                name="x",
                data_types=[mlc.DataType.FLOAT],
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="x")
                )
            ),
            mlc.Field(
                id="transient_candidates/y",
                name="y",
                data_types=[mlc.DataType.FLOAT],
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="y")
                )
            ),
            mlc.Field(
                id="transient_candidates/flux",
                name="flux",
                data_types=[mlc.DataType.FLOAT],
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="flux")
                )
            ),
            
            mlc.Field(
                id="transient_candidates/full_image_path",
                name="full_image_path",
                description="Relative path to the .npy file containing the (4096,4096,4) tensor",
                data_types=[mlc.DataType.TEXT], 
                source=mlc.Source(
                    file_object="master_index",
                    extract=mlc.Extract(column="image_filename")
                )
            )
        ]
    )
]

metadata = mlc.Metadata(
    name="roman_croissant_full_images",
    description="",
    distribution=distribution,
    record_sets=record_sets
)

output_path = os.path.join(BASE_DIR, "croissant.json")
with open(output_path, "w") as f:
    f.write(json.dumps(metadata.to_json(), indent=2))

print(f"Successfully generated {output_path}")