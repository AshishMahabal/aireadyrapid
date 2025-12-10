# Croissant Cutouts Pipeline

This folder contains scripts to create, document, and test a machine learning dataset of 64x64 cutouts for transient detection using the [Croissant](https://github.com/mlcommons/croissant) metadata format.

## Overview

The pipeline processes astronomical FITS images from job folders, extracts 64x64 cutouts centered on each candidate position, and creates a Croissant-compliant dataset. The pipeline includes **automatic injection tracking** via on-the-fly truth table generation and cross-matching. Unlike the full images pipeline, this approach pre-extracts cutouts during dataset creation, resulting in faster training times and reduced memory usage.

### Key Features:
- **Automatic Truth Generation**: Generates truth tables and cross-matching on-the-fly from source catalogs
- **Injection Tracking**: Identifies artificially injected vs natural transients
- **Truth Linking**: Connects detections to ground truth via cross-matching
- **Pre-extracted Cutouts**: 64x64x4 tensors for fast training
- **No Pre-processing Required**: Directly processes raw RAPID outputs without intermediate CSV files

## Scripts

### 1. `create_dataset.py`

Processes raw FITS files from job folders and creates a dataset of 64x64 cutouts with automatic truth table generation.

**What it does:**
- Reads 4 FITS files per job (science, reference, difference, SCORR images)
- Applies ZScale normalization to each full image
- **Automatically generates truth tables** from injection catalogs and OpenUniverse catalogs
- **Cross-matches detections** with truth sources (4 pixel radius)
- Extracts injection status from truth catalogs
- For each candidate in the catalog, extracts a 64x64 cutout from each normalized image
- Stacks the 4 cutouts into a (64, 64, 4) tensor and saves as `.npy` files
- Creates a `master_index.csv` with all candidates, labels, and truth linking

**Usage:**
```bash
python create_dataset.py --input_dir <path_to_jobs> --output_dir <output_path>
```

**Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `./mini_dataset` | Input directory containing `jid*` job folders |
| `--output_dir` | `-o` | `./hackathon_dataset` | Output directory for processed dataset |

**Expected input structure:**
```
input_dir/
├── jid001/
│   ├── bkg_subbed_science_image.fits
│   ├── awaicgen_output_mosaic_image_resampled_gainmatched.fits
│   ├── diffimage_masked.fits
│   ├── scorrimage_masked.fits
│   ├── diffimage_masked_psfcat_finder.txt      # Finder catalog
│   ├── diffimage_masked_psfcat.txt             # PSF photometry
│   ├── Roman_TDS_simple_model_*.txt            # Injection truth catalogs (auto-detected)
│   ├── Roman_TDS_index_*.txt                   # OpenUniverse catalogs (auto-detected)
│   └── Roman_TDS_simple_model_*_reformatted.fits  # Science image with metadata
├── jid002/
│   └── ...
```

**Output structure:**
```
output_dir/
├── cutouts/
│   ├── jid001_cutout00000.npy
│   ├── jid001_cutout00001.npy
│   ├── jid002_cutout00000.npy
│   └── ...
└── master_index.csv
```

**Truth Table Generation:**
The pipeline automatically:
1. Searches for injection catalogs matching `Roman_TDS_simple_model_([a-zA-Z])(\d+)_(\d+)_(\d+)_lite_inject\.txt`
2. Searches for OpenUniverse catalogs matching `Roman_TDS_index_([a-zA-Z])(\d+)_(\d+)_(\d+)\.txt`
3. Extracts filter and zero-point magnitude from science image FITS header
4. Combines all truth sources with injection flags
5. Cross-matches PSF detections within 4 pixels
6. Assigns `match_id` to matched sources (format: `18_inj` for injections, `20149058_ou` for OpenUniverse)
7. Labels: Real (1) if matched, Bogus (0) if unmatched

---

### 2. `generate_croissant.py`

Generates a Croissant metadata file (`croissant.json`) for the cutout dataset with injection tracking fields.

**What it does:**
- Reads the master index CSV
- Documents injection status, truth linking fields
- Creates Croissant-compliant JSON metadata describing the dataset schema

**Usage:**
```bash
python generate_croissant.py --csv_path <path_to_csv> --output <output_json_path>
```

**Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--csv_path` | `-c` | `./hackathon_dataset/master_index.csv` | Path to the master index CSV file |
| `--output` | `-o` | `./hackathon_dataset/croissant.json` | Output path for `croissant.json` |

---

### 3. `test_croissant_dataset.py`

Tests the dataset by training a CNN classifier.

**What it does:**
- Loads the dataset using Croissant metadata
- Directly loads pre-extracted 64x64 cutouts
- Combines image data with tabular features (sharpness, roundness, flux, magnitude, etc.)
- Trains a CNN for binary classification

**Usage:**
```bash
python test_croissant_dataset.py --dataset_dir <path_to_dataset>
```

**Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--dataset_dir` | `-d` | `./hackathon_dataset` | Dataset directory containing `croissant.json` |

---

## Full Pipeline Example

```bash
# Step 1: Create the cutout dataset from raw FITS files
python create_dataset.py -i ./raw_data -o ./my_hackathon_dataset

# Step 2: Generate Croissant metadata
python generate_croissant.py -c ./my_hackathon_dataset/master_index.csv -o ./my_hackathon_dataset/croissant.json

# Step 3: Test the dataset with a simple training run
python test_croissant_dataset.py -d ./my_hackathon_dataset
```

Install dependencies:
```bash
pip install numpy pandas astropy mlcroissant torch
```

## Dataset Schema

The `master_index.csv` contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | float | Candidate ID from finder catalog |
| `cutout_id` | int | Unique cutout identifier |
| `jid` | string | Job identifier |
| `x` | float | X coordinate in original image |
| `y` | float | Y coordinate in original image |
| `sharpness` | float | Source sharpness |
| `roundness1` | float | First roundness metric |
| `roundness2` | float | Second roundness metric |
| `npix` | float | Number of pixels |
| `peak` | float | Peak pixel value |
| `flux` | float | Measured flux |
| `mag` | float | Instrumental magnitude |
| `daofind_mag` | float | DAOFind magnitude |
| `flags` | float | Quality flags from psfcat |
| `match_id` | string | Match ID linking to truth table (e.g., '18_inj', '20149058_ou') |
| `truth_id` | string | Truth table ID for matched sources |
| `injected` | bool | Whether source was artificially injected |
| `label` | int | Binary label (0=bogus, 1=real) |
| `cutout_filename` | string | Relative path to the `.npy` cutout file |

## Cutout Tensor Format

Each `.npy` file contains a tensor of shape `(64, 64, 4)` where the 4 channels are:
1. **Science image** - Background-subtracted science frame cutout
2. **Reference image** - Resampled reference/template image cutout
3. **Difference image** - Science minus reference cutout (masked)
4. **SCORR image** - SCORR map cutout (masked)

ZScale normalization is applied to each full image before cutout extraction, and cutouts near image edges are zero-padded to maintain the 64x64 size.
