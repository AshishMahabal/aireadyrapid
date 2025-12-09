import mlcroissant as mlc
import json
import hashlib
import os
import argparse

def get_sha256(filepath):
    """Calculate SHA256 hash for a file"""
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath} to calculate hash.")
        return ""

def create_field_list(file_object_id, prefix):
    """Create list of fields for a record set"""
    return [
        mlc.Field(
            id=f"{prefix}/id",
            name="id",
            description="Candidate identifier from source finder",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="id")
            )
        ),
        mlc.Field(
            id=f"{prefix}/x",
            name="x",
            description="Pixel x-coordinate",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="x")
            )
        ),
        mlc.Field(
            id=f"{prefix}/y",
            name="y",
            description="Pixel y-coordinate",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="y")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sharpness",
            name="sharpness",
            description="Source sharpness metric",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sharpness")
            )
        ),
        mlc.Field(
            id=f"{prefix}/roundness1",
            name="roundness1",
            description="First roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="roundness1")
            )
        ),
        mlc.Field(
            id=f"{prefix}/roundness2",
            name="roundness2",
            description="Second roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="roundness2")
            )
        ),
        mlc.Field(
            id=f"{prefix}/npix",
            name="npix",
            description="Number of pixels above threshold",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="npix")
            )
        ),
        mlc.Field(
            id=f"{prefix}/peak",
            name="peak",
            description="Peak pixel intensity",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="peak")
            )
        ),
        mlc.Field(
            id=f"{prefix}/flux",
            name="flux",
            description="Measured source flux",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="flux")
            )
        ),
        mlc.Field(
            id=f"{prefix}/mag",
            name="mag",
            description="Instrumental magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/daofind_mag",
            name="daofind_mag",
            description="DAOFind-derived magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="daofind_mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/flags",
            name="flags",
            description="Quality flags from photometry",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="flags")
            )
        ),
        mlc.Field(
            id=f"{prefix}/match",
            name="match",
            description="Cross-match indicator (-1 for bogus)",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="match")
            )
        ),
        mlc.Field(
            id=f"{prefix}/label",
            name="label",
            description="Binary classification (0=bogus, 1=real)",
            data_types=[mlc.DataType.INTEGER],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="label")
            )
        ),
        mlc.Field(
            id=f"{prefix}/jid",
            name="jid",
            description="Job identifier",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="jid")
            )
        ),
        mlc.Field(
            id=f"{prefix}/image_path",
            name="image_path",
            description="Path to 7-channel image tensor (.npy)",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="image_filename")
            )
        )
    ]

def generate_croissant(dataset_dir, output_path):
    zogy_csv_path = os.path.join(dataset_dir, "zogy_candidates.csv")
    sfft_csv_path = os.path.join(dataset_dir, "sfft_candidates.csv")
    
    zogy_sha256 = get_sha256(zogy_csv_path)
    sfft_sha256 = get_sha256(sfft_csv_path)
    
    print(f"Calculated SHA256 for {zogy_csv_path}: {zogy_sha256}")
    print(f"Calculated SHA256 for {sfft_csv_path}: {sfft_sha256}")

    ctx = mlc.Context()

    distribution = [
        mlc.FileObject(
            id="zogy_index",
            name="zogy_index",
            content_url="zogy_candidates.csv",
            encoding_formats=["text/csv"],
            sha256=zogy_sha256
        ),
        mlc.FileObject(
            id="sfft_index",
            name="sfft_index",
            content_url="sfft_candidates.csv",
            encoding_formats=["text/csv"],
            sha256=sfft_sha256
        ),
        mlc.FileSet(
            id="npy_images",
            name="npy_images",
            includes="images/*.npy", 
            encoding_formats=["application/x-numpy"]
        ),
        mlc.FileSet(
            id="catalog_files",
            name="catalog_files",
            includes="catalogs/*",
            encoding_formats=["text/plain", "image/fits"]
        )
    ]

    record_sets = [
        mlc.RecordSet(
            id="zogy_candidates",
            name="zogy_candidates",
            fields=create_field_list("zogy_index", "zogy_candidates")
        ),
        mlc.RecordSet(
            id="sfft_candidates",
            name="sfft_candidates",
            fields=create_field_list("sfft_index", "sfft_candidates")
        )
    ]

    metadata = mlc.Metadata(
        name="roman_croissant_difference_imaging",
        description="Difference imaging dataset with separate ZOGY and SFFT candidate record sets, catalogs, and metadata for transient classification",
        distribution=distribution,
        record_sets=record_sets
    )

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata.to_json(), indent=2))

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Croissant metadata file")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./hackathon_dataset",
                        help="Dataset directory containing CSV files (default: ./hackathon_dataset)")
    parser.add_argument("--output", "-o", type=str, default="./hackathon_dataset/croissant.json",
                        help="Output path for croissant.json (default: ./hackathon_dataset/croissant.json)")
    args = parser.parse_args()
    
    generate_croissant(args.dataset_dir, args.output)
