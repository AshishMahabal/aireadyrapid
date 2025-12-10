# Croissant Full Images Pipeline

This folder contains scripts to create, document, and test an AI ready dataset for transient detection using the [Croissant](https://github.com/mlcommons/croissant) metadata format.

## Overview

The pipeline processes astronomical FITS images from job folders, creates a unified dataset with 4-channel tensors (science, reference, difference, and SCORR images), and generates Croissant-compliant metadata for AI-ready data loading. **The pipeline now includes automatic injection tracking** via on-the-fly truth table generation and cross-matching.

### Key Features:
- **Automatic Truth Generation**: Generates truth tables and cross-matching on-the-fly from source catalogs
- **Injection Tracking**: Identifies artificially injected vs natural transients
- **Truth Linking**: Connects detections to ground truth via cross-matching
- **Full Resolution Images**: 4090x4090 4-channel tensors for flexible analysis
- **No Pre-processing Required**: Directly processes raw RAPID outputs without intermediate CSV files

## Scripts

### 1. `create_dataset.py`

Processes raw FITS files from job folders and creates a normalized dataset with automatic truth table generation.

**What it does:**
- Reads 4 FITS files per job (science, reference, difference, SCORR images)
- Applies ZScale normalization to each image
- Stacks them into a 4-channel tensor and saves as `.npy` files
- **Automatically generates truth tables** from injection catalogs and OpenUniverse catalogs
- **Cross-matches detections** with truth sources (4 pixel radius)
- Extracts injection status from truth catalogs
- Creates a `master_index.csv` with all candidates, labels, and truth linking

**Truth Table Generation:**
The pipeline automatically:
1. Searches for injection catalogs matching `Roman_TDS_simple_model_([a-zA-Z])(\d+)_(\d+)_(\d+)_lite_inject\.txt`
2. Searches for OpenUniverse catalogs matching `Roman_TDS_index_([a-zA-Z])(\d+)_(\d+)_(\d+)\.txt`
3. Extracts filter and zero-point magnitude from science image FITS header
4. Combines all truth sources with injection flags
5. Cross-matches PSF detections within 4 pixels (MAG_LIM=26.0)
6. Assigns `match_id` to matched sources (format: `18_inj` for injections, `20149058_ou` for OpenUniverse)
7. Labels: Real (1) if matched, Bogus (0) if unmatched

**Usage:**
```bash
# Basic usage with default directories
python create_dataset.py

# Specify custom directories
python create_dataset.py --input_dir /path/to/jobs --output_dir /path/to/output

# Filter by specific fields from field list files
python create_dataset.py --input_dir /path/to/jobs --output_dir /path/to/output \
  --field_files H158_fields.txt R062_fields.txt \
  --fields 5261331 5325281 5356473

# Set MJD upper bound
python create_dataset.py --input_dir /path/to/jobs --output_dir /path/to/output \
  --field_files R062_fields.txt --mjd_max 62100.0

# Combine all filters
python create_dataset.py --input_dir /path/to/jobs --output_dir /path/to/output \
  --field_files H158_fields.txt R062_fields.txt \
  --fields 5261331 5297552 --mjd_max 62075.5
```

**Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `./mini_dataset` | Input directory containing `jid*` job folders |
| `--output_dir` | `-o` | `./hackathon_dataset` | Output directory for processed dataset |
| `--field_files` | | None | Field list text files (e.g., H158_fields.txt R062_fields.txt) |
| `--fields` | | None | Specific field IDs to process (space-separated integers) |
| `--mjd_max` | | None | Maximum MJD value (upper bound, accepts decimals) |

**Field List Filtering:**
When `--field_files` is provided, the script will:
- Read job IDs from the specified field list files
- Filter by `--fields` (if provided) to process only specific field IDs
- Filter by `--mjd_max` (if provided) to process only observations up to that MJD
- If no field files are provided, the script discovers all `jid*` folders automatically

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
├── images/
│   ├── jid001.npy
│   ├── jid002.npy
│   └── ...
└── master_index.csv
```

---

### 2. `generate_croissant.py`

Generates a Croissant metadata file (`croissant.json`) for the dataset with injection tracking fields.

**What it does:**
- Reads the master index CSV
- Creates Croissant JSON metadata describing the dataset

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
- Extracts 64x64 cutouts around each candidate position
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
# Step 1: Create the dataset from raw FITS files
python create_dataset.py -i ./raw_data -o ./my_dataset

# Step 2: Generate Croissant metadata
python generate_croissant.py -c ./my_dataset/master_index.csv -o ./my_dataset/croissant.json

# Step 3: Test the dataset with a simple training run
python test_croissant_dataset.py -d ./my_dataset
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
| `jid` | string | Job identifier |
| `x` | float | X coordinate in image |
| `y` | float | Y coordinate in image |
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
| `image_filename` | string | Relative path to the `.npy` image tensor |

## Image Tensor Format

Each `.npy` file contains a tensor of shape `(H, W, 4)` where the 4 channels are:
1. **Science image** - Background-subtracted science frame
2. **Reference image** - Resampled reference/template image
3. **Difference image** - Science minus reference (masked)
4. **SCORR image** - SCORR map (masked)

All channels are ZScale normalized to the range [0, 1].
