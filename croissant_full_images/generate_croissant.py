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
                    id="transient_candidates/id",
                    name="id",
                    description="Candidate identifier from source finder",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="id")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/x",
                    name="x",
                    description="Pixel x-coordinate",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="x")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/y",
                    name="y",
                    description="Pixel y-coordinate",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="y")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/sharpness",
                    name="sharpness",
                    description="Source sharpness metric",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="sharpness")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/roundness1",
                    name="roundness1",
                    description="First roundness parameter",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="roundness1")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/roundness2",
                    name="roundness2",
                    description="Second roundness parameter",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="roundness2")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/npix",
                    name="npix",
                    description="Number of pixels above threshold",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="npix")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/peak",
                    name="peak",
                    description="Peak pixel intensity",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="peak")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/flux",
                    name="flux",
                    description="Measured source flux",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="flux")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/mag",
                    name="mag",
                    description="Instrumental magnitude",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="mag")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/daofind_mag",
                    name="daofind_mag",
                    description="DAOFind-derived magnitude",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="daofind_mag")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/flags",
                    name="flags",
                    description="Quality flags from photometry",
                    data_types=[mlc.DataType.FLOAT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="flags")
                    )
                ),
                
                # Truth and metadata fields
                mlc.Field(
                    id="transient_candidates/match_id",
                    name="match_id",
                    description="Match ID linking to truth table (empty for bogus, e.g., '18_inj', '20149058_ou')",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="match_id")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/truth_id",
                    name="truth_id",
                    description="Truth table ID for matched sources (e.g., '18_inj', '20149058_ou')",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="truth_id")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/injected",
                    name="injected",
                    description="Whether source was artificially injected (True/False)",
                    data_types=[mlc.DataType.BOOL],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="injected")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/label",
                    name="label",
                    description="Binary classification (0=bogus, 1=real)",
                    data_types=[mlc.DataType.INTEGER],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="label")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/jid",
                    name="jid",
                    description="Job identifier",
                    data_types=[mlc.DataType.TEXT],
                    source=mlc.Source(
                        file_object="master_index",
                        extract=mlc.Extract(column="jid")
                    )
                ),
                mlc.Field(
                    id="transient_candidates/image_path",
                    name="image_path",
                    description="Path to 4-channel image tensor (.npy)",
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
        description="Transient detection dataset with full resolution 4-channel images (science, reference, difference, score) and candidate metadata with injection tracking for real/bogus classification",
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