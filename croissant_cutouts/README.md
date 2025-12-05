# Croissant Cutouts Pipeline

This folder contains scripts to create, document, and test a machine learning dataset of 64x64 cutouts for transient detection using the [Croissant](https://github.com/mlcommons/croissant) metadata format.

## Overview

The pipeline processes astronomical FITS images from job folders, extracts 64x64 cutouts centered on each candidate position, and creates a Croissant-compliant dataset. Unlike the full images pipeline, this approach pre-extracts cutouts during dataset creation, resulting in faster training times and reduced memory usage.

## Scripts

### 1. `create_dataset.py`

Processes raw FITS files from job folders and creates a dataset of 64x64 cutouts.

**What it does:**
- Reads 4 FITS files per job (science, reference, difference, score images)
- Applies ZScale normalization to each full image
- For each candidate in the catalog, extracts a 64x64 cutout from each normalized image
- Stacks the 4 cutouts into a (64, 64, 4) tensor and saves as `.npy` files
- Creates a `master_index.csv` with all candidates and their labels

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
│   ├── awaicgen_output_mosaic_image_resampled.fits
│   ├── diffimage_masked.fits
│   ├── scorrimage_masked.fits
│   └── psfcat.csv
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

---

### 2. `generate_croissant.py`

Generates a Croissant metadata file (`croissant.json`) for the cutout dataset.

**What it does:**
- Reads the master index CSV
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

Tests the Croissant dataset by training a simple CNN classifier.

**What it does:**
- Loads the dataset using the Croissant metadata
- Directly loads pre-extracted 64x64 cutouts
- Trains a simple CNN for binary classification (real vs bogus transients)

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
| `cutout_id` | int | Unique cutout identifier |
| `job_id` | string | Original job identifier |
| `x` | float | X coordinate in original image |
| `y` | float | Y coordinate in original image |
| `label` | int | Binary label (0=bogus, 1=real) |
| `cutout_filename` | string | Relative path to the `.npy` cutout file |
| `flux` | float | Measured flux of candidate (if available) |
| `fwhm` | float | Full-width half-maximum (if available) |
| `elongation` | float | Source elongation (if available) |

## Cutout Tensor Format

Each `.npy` file contains a tensor of shape `(64, 64, 4)` where the 4 channels are:
1. **Science image** - Background-subtracted science frame cutout
2. **Reference image** - Resampled reference/template image cutout
3. **Difference image** - Science minus reference cutout (masked)
4. **Score image** - Significance/score map cutout (masked)

ZScale normalization is applied to each full image before cutout extraction, and cutouts near image edges are zero-padded to maintain the 64x64 size.
