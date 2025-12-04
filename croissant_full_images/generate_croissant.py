import mlcroissant as mlc
import json
import hashlib
import os
import argparse

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

def generate_croissant(csv_path, output_path):
    csv_sha256 = get_sha256(csv_path)
    print(f"Calculated SHA256 for {csv_path}: {csv_sha256}")
    
    csv_filename = os.path.basename(csv_path)

    ctx = mlc.Context()

    distribution = [
        mlc.FileObject(
            id="master_index",
            name="master_index",
            content_url=csv_filename,
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

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata.to_json(), indent=2))

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Croissant metadata file")
    parser.add_argument("--csv_path", "-c", type=str, default="./hackathon_dataset/master_index.csv",
                        help="Path to the master index CSV file (default: ./hackathon_dataset/master_index.csv)")
    parser.add_argument("--output", "-o", type=str, default="./hackathon_dataset/croissant.json",
                        help="Output path for croissant.json (default: ./hackathon_dataset/croissant.json)")
    args = parser.parse_args()
    
    generate_croissant(args.csv_path, args.output)