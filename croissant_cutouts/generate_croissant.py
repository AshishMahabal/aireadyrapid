import mlcroissant as mlc
import json
import hashlib
import os
import argparse

# calculating the SHA256 sum (required for FileObject)
def get_sha256(filepath):
    """Calculate SHA256 hash of a file."""
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

    distribution = [
        mlc.FileObject(
            id="master_index",
            name="master_index",
            content_url=csv_filename,
            encoding_formats=["text/csv"],
            sha256=csv_sha256
        ),
        mlc.FileSet(
            id="npy_cutouts",
            name="npy_cutouts",
            includes="cutouts/*.npy",
            encoding_formats=["application/x-numpy"]
        )
    ]

    record_sets = [
        mlc.RecordSet(
            id="transient_candidates",
            name="transient_candidates",
            fields=[
                mlc.Field(
                    id="transient_candidates/id",
                    name="id",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="id")
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
                    id="transient_candidates/sharpness",
                    name="sharpness",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="sharpness")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/roundness1",
                    name="roundness1",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="roundness1")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/roundness2",
                    name="roundness2",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="roundness2")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/npix",
                    name="npix",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="npix")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/peak",
                    name="peak",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="peak")
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
                    id="transient_candidates/mag",
                    name="mag",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="mag")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/daofind_mag",
                    name="daofind_mag",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="daofind_mag")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/flags",
                    name="flags",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="flags")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/match",
                    name="match",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="match")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/cutout_id",
                    name="cutout_id",
                    description="Unique cutout identifier",
                    data_types=[mlc.DataType.INTEGER],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="cutout_id")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/jid",
                    name="jid",
                    description="Job ID",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="jid")
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
                    id="transient_candidates/cutout_path",
                    name="cutout_path",
                    description="Relative path to the .npy file containing the (64,64,4) cutout tensor",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="cutout_filename")
                    )
                )
            ]
        )
    ]

    metadata = mlc.Metadata(
        name="roman_croissant_cutouts",
        description="64x64 cutouts of the transient candidates.",
        distribution=distribution,
        record_sets=record_sets
    )

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata.to_json(), indent=2))

    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Croissant metadata file for cutout dataset")
    parser.add_argument("--csv_path", "-c", type=str, default="./hackathon_dataset/master_index.csv",
                        help="Path to the master index CSV file (default: ./hackathon_dataset/master_index.csv)")
    parser.add_argument("--output", "-o", type=str, default="./hackathon_dataset/croissant.json",
                        help="Output path for croissant.json (default: ./hackathon_dataset/croissant.json)")
    args = parser.parse_args()
    
    generate_croissant(args.csv_path, args.output)
