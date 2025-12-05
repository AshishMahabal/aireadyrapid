# AI-Ready RAPID

Tools for creating AI-ready datasets from Nancy Grace Roman Space Telescope transient detection pipeline outputs.

This repository provides two pipelines for preparing [Croissant](https://github.com/mlcommons/croissant)-compliant datasets from astronomical FITS images, enabling machine learning workflows for real/bogus transient classification.

## Overview

The pipelines process difference imaging outputs from the Roman telescope's transient detection system. Each job folder contains:
- Science image
- Reference image
- Difference image
- SCORR image
- Candidate catalog with positions and metadata

## Pipelines

### 1. Full Images Pipeline (`croissant_full_images/`)

Stores full resolution images and extracts cutouts at training time.

[View full documentation](croissant_full_images/README.md)

### 2. Cutouts Pipeline (`croissant_cutouts/`)

Pre-extracts 64x64 cutouts centered on each candidate during dataset creation.

[View full documentation](croissant_cutouts/README.md)

## Usage

### Installation

```bash
pip install numpy pandas astropy mlcroissant torch
```

### Using the Full Images Pipeline

```bash
cd croissant_full_images

# Step 1: Create dataset
python create_dataset.py -i /path/to/raw_data -o ./dataset

# Step 2: Generate Croissant metadata
python generate_croissant.py -c ./dataset/master_index.csv -o ./dataset/croissant.json

# Step 3: Test with training
python test_croissant_dataset.py -d ./dataset
```

### Using the Cutouts Pipeline

```bash
cd croissant_cutouts

# Step 1: Create dataset
python create_dataset.py -i /path/to/raw_data -o ./dataset

# Step 2: Generate Croissant metadata
python generate_croissant.py -c ./dataset/master_index.csv -o ./dataset/croissant.json

# Step 3: Test with training
python test_croissant_dataset.py -d ./dataset
```

## Input Data Format

Both pipelines expect the following input structure:

```
input_dir/
├── jid001/
│   ├── bkg_subbed_science_image.fits
│   ├── awaicgen_output_mosaic_image_resampled.fits
│   ├── diffimage_masked.fits
│   ├── scorrimage_masked.fits
│   └── psfcat.csv
├── jid002/
│   └── ...
```

The `psfcat.csv` catalog should contain at minimum:
- `x`, `y`: Candidate pixel coordinates
- `match`: Match indicator (-1 = bogus, other = real)

## Output Format

Both pipelines produce:
- `.npy` files containing 4-channel tensors (normalized to 0-1 via ZScale)
- `master_index.csv` with candidate metadata and labels
- `croissant.json` for Croissant-compliant data loading

## License

Apache License 2.0
