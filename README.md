# AI-Ready RAPID

Tools for creating AI-ready datasets from Nancy Grace Roman Space Telescope transient detection pipeline outputs.

This repository provides three pipelines for preparing [Croissant](https://github.com/mlcommons/croissant)-compliant datasets from astronomical FITS images, enabling machine learning workflows for real/bogus transient classification.

## Overview

The pipelines process difference imaging outputs from the Roman telescope's transient detection system. All pipelines include:
- **Automatic truth table generation** from injection and OpenUniverse catalogs
- **Cross-matching** with 4-pixel radius for real/bogus classification
- **Injection tracking** to distinguish artificially injected from natural transients
- **No pre-processing required** - processes raw RAPID outputs directly

Each job folder contains:
- Science image
- Reference image  
- Difference image(s)
- SCORR image(s)
- PSF catalogs with detections
- Truth catalogs (injection + OpenUniverse)

## Pipelines

### 1. Difference Imaging Pipeline (`croissant_difference_imaging/`)

**Unified ZOGY + SFFT comparison dataset** with 9-channel images.

Combines both ZOGY and SFFT difference imaging algorithms in a single dataset for direct algorithm comparison. Each candidate has features from both methods side-by-side.

[View full documentation](croissant_difference_imaging/README.md)

### 2. Full Images Pipeline (`croissant_full_images/`)

**Full resolution 4-channel images** with on-demand cutout extraction.

Stores full resolution images (4090x4090) and extracts cutouts at training time. Best for flexible analysis requiring access to surrounding context.

[View full documentation](croissant_full_images/README.md)

### 3. Cutouts Pipeline (`croissant_cutouts/`)

**Pre-extracted 64x64 cutouts** for fast training.

Pre-extracts 64x64 cutouts centered on each candidate during dataset creation. Best for rapid iteration and reduced memory usage during training.

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

### Using the Difference Imaging Pipeline

```bash
cd croissant_difference_imaging

# Step 1: Create dataset (with optional algorithm selection)
python create_dataset.py -i /path/to/raw_data -o ./dataset --algorithm both

# Step 2: Generate Croissant metadata
python generate_croissant.py -c ./dataset/candidates.csv -o ./dataset/croissant.json

# Step 3: Test with training
python test_croissant_dataset.py -d ./dataset
```

## Input Data Format

All pipelines expect job folders with FITS images and catalogs:

```
input_dir/
├── jid001/
│   ├── bkg_subbed_science_image.fits
│   ├── awaicgen_output_mosaic_image_resampled_gainmatched.fits
│   ├── diffimage_masked.fits
│   ├── scorrimage_masked.fits
│   ├── diffimage_masked_psfcat_finder.txt      # Finder catalog
│   ├── diffimage_masked_psfcat.txt             # PSF photometry
│   ├── Roman_TDS_simple_model_*_lite_inject.txt    # Injection catalogs (auto-detected)
│   ├── Roman_TDS_index_*.txt                       # OpenUniverse catalogs (auto-detected)
│   └── Roman_TDS_simple_model_*_reformatted.fits   # Science image with metadata
├── jid002/
│   └── ...
```

**Additional files for difference_imaging pipeline:**
- `sfftdiffimage_dconv_masked.fits`, `sfftdiffimage_cconv_masked.fits` (SFFT difference images)
- `diffpsf.fits` (PSF image)
- `diffimage_uncert_masked.fits`, `sfftdiffimage_uncert_masked.fits` (Uncertainty maps)
- SFFT catalogs: `sfftdiffimage_masked_psfcat_finder.txt`, `sfftdiffimage_masked_psfcat.txt`

**Truth Catalogs (all pipelines):**
- Injection catalogs are automatically detected by regex pattern
- OpenUniverse catalogs provide natural transient ground truth
- Cross-matching performed with 4-pixel radius and MAG_LIM=26.0

## Output Format

All pipelines produce:
- `.npy` files containing image tensors (ZScale normalized to 0-1)
  - Full images: 4-channel (4090x4090)
  - Cutouts: 4-channel (64x64)
  - Difference imaging: 9-channel (4090x4090)
- CSV with candidate metadata and labels:
  - `match_id`: Truth match ID (e.g., '18_inj', '20149058_ou')
  - `truth_id`: Same as match_id
  - `injected`: Boolean flag (True=injected, False=natural)
  - `label`: Binary classification (0=bogus, 1=real)
- `croissant.json` for Croissant-compliant data loading

## Key Features

- **Zero pre-processing**: No intermediate CSV files required - generates truth tables on-the-fly
- **Injection tracking**: Automatically identifies 73 injected sources across datasets
- **Consistent labeling**: ~79 real transients identified via cross-matching
- **Croissant-compliant**: Proper TEXT/BOOL data types prevent loading errors
- **Algorithm comparison**: Difference imaging pipeline enables ZOGY vs SFFT analysis

## License

Apache License 2.0
