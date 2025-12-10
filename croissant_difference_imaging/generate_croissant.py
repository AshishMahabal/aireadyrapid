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
    """Create list of fields for candidate record set"""
    return [
        # Core identification and position
        mlc.Field(
            id=f"{prefix}/id",
            name="id",
            description="Candidate identifier",
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
        
        # Truth and metadata
        mlc.Field(
            id=f"{prefix}/match_id",
            name="match_id",
            description="Match ID linking to truth table (empty for bogus, e.g., '18_inj', '20149058_ou')",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="match_id")
            )
        ),
        mlc.Field(
            id=f"{prefix}/truth_id",
            name="truth_id",
            description="Truth table ID for matched sources (e.g., '18_inj', '20149058_ou')",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="truth_id")
            )
        ),
        mlc.Field(
            id=f"{prefix}/injected",
            name="injected",
            description="Whether source was artificially injected (True/False)",
            data_types=[mlc.DataType.BOOL],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="injected")
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
        
        # ZOGY-specific features
        mlc.Field(
            id=f"{prefix}/zogy_sharpness",
            name="zogy_sharpness",
            description="ZOGY: Source sharpness metric",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_sharpness")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_roundness1",
            name="zogy_roundness1",
            description="ZOGY: First roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_roundness1")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_roundness2",
            name="zogy_roundness2",
            description="ZOGY: Second roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_roundness2")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_npix",
            name="zogy_npix",
            description="ZOGY: Number of pixels above threshold",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_npix")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_peak",
            name="zogy_peak",
            description="ZOGY: Peak pixel intensity",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_peak")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_flux",
            name="zogy_flux",
            description="ZOGY: Measured source flux",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_flux")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_mag",
            name="zogy_mag",
            description="ZOGY: Instrumental magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_daofind_mag",
            name="zogy_daofind_mag",
            description="ZOGY: DAOFind-derived magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_daofind_mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/zogy_flags",
            name="zogy_flags",
            description="ZOGY: Quality flags from photometry",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="zogy_flags")
            )
        ),
        
        # SFFT-specific features
        mlc.Field(
            id=f"{prefix}/sfft_sharpness",
            name="sfft_sharpness",
            description="SFFT: Source sharpness metric",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_sharpness")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_roundness1",
            name="sfft_roundness1",
            description="SFFT: First roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_roundness1")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_roundness2",
            name="sfft_roundness2",
            description="SFFT: Second roundness parameter",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_roundness2")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_npix",
            name="sfft_npix",
            description="SFFT: Number of pixels above threshold",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_npix")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_peak",
            name="sfft_peak",
            description="SFFT: Peak pixel intensity",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_peak")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_flux",
            name="sfft_flux",
            description="SFFT: Measured source flux",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_flux")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_mag",
            name="sfft_mag",
            description="SFFT: Instrumental magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_daofind_mag",
            name="sfft_daofind_mag",
            description="SFFT: DAOFind-derived magnitude",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_daofind_mag")
            )
        ),
        mlc.Field(
            id=f"{prefix}/sfft_flags",
            name="sfft_flags",
            description="SFFT: Quality flags from photometry",
            data_types=[mlc.DataType.FLOAT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="sfft_flags")
            )
        ),
        
        # Image reference
        mlc.Field(
            id=f"{prefix}/image_path",
            name="image_path",
            description="Path to 9-channel image tensor (.npy) with ZOGY and SFFT products",
            data_types=[mlc.DataType.TEXT],
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(column="image_filename")
            )
        )
    ]

def generate_croissant(dataset_dir, output_path):
    candidates_csv_path = os.path.join(dataset_dir, "candidates.csv")
    
    candidates_sha256 = get_sha256(candidates_csv_path)
    
    print(f"Calculated SHA256 for {candidates_csv_path}: {candidates_sha256}")

    ctx = mlc.Context()

    distribution = [
        mlc.FileObject(
            id="candidates_index",
            name="candidates_index",
            content_url="candidates.csv",
            encoding_formats=["text/csv"],
            sha256=candidates_sha256
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
            encoding_formats=["text/plain"]
        )
    ]

    record_sets = [
        mlc.RecordSet(
            id="transient_candidates",
            name="transient_candidates",
            fields=create_field_list("candidates_index", "transient_candidates")
        )
    ]

    metadata = mlc.Metadata(
        name="roman_croissant_difference_imaging",
        description="Unified difference imaging dataset with 9-channel image tensors (science, reference, ZOGY diff/SCORR, SFFT diff/SCORR, PSF, uncertainties), combined ZOGY+SFFT features per candidate, and injection tracking for comprehensive transient classification and algorithm comparison",
        cite_as="December 2025 Roman Quarterly",
        version="0.9.0",
        date_published="2025-12-10",
        license=["https://creativecommons.org/licenses/by/4.0/"],
        distribution=distribution,
        record_sets=record_sets
    )

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata.to_json(), indent=2, default=str))

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Croissant metadata file")
    parser.add_argument("--dataset_dir", "-d", type=str, default="./hackathon_dataset",
                        help="Dataset directory containing CSV files (default: ./hackathon_dataset)")
    parser.add_argument("--output", "-o", type=str, default="./hackathon_dataset/croissant.json",
                        help="Output path for croissant.json (default: ./hackathon_dataset/croissant.json)")
    args = parser.parse_args()
    
    generate_croissant(args.dataset_dir, args.output)
