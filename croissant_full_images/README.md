# Croissant Full Images Pipeline

This folder contains scripts to create, document, and test an AI ready dataset for transient detection using the [Croissant](https://github.com/mlcommons/croissant) metadata format.

## Overview

The pipeline processes astronomical FITS images from job folders, creates a unified dataset with 4-channel tensors (science, reference, difference, and score images), and generates Croissant-compliant metadata for AI-ready data loading.

## Scripts

### 1. `create_dataset.py`

Processes raw FITS files from job folders and creates a normalized dataset.

**What it does:**
- Reads 4 FITS files per job (science, reference, difference, score images)
- Applies ZScale normalization to each image
- Stacks them into a 4-channel tensor and saves as `.npy` files
- Extracts candidate information from `psfcat.csv` catalogs
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
├── images/
│   ├── jid001.npy
│   ├── jid002.npy
│   └── ...
└── master_index.csv
```

---

### 2. `generate_croissant.py`

Generates a Croissant metadata file (`croissant.json`) for the dataset.

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
| `match` | float | Match indicator from psfcat |
| `label` | int | Binary label (0=bogus, 1=real) |
| `image_filename` | string | Relative path to the `.npy` image tensor |

## Image Tensor Format

Each `.npy` file contains a tensor of shape `(H, W, 4)` where the 4 channels are:
1. **Science image** - Background-subtracted science frame
2. **Reference image** - Resampled reference/template image
3. **Difference image** - Science minus reference (masked)
4. **Score image** - Significance/score map (masked)

All channels are ZScale normalized to the range [0, 1].
