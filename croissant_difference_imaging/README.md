# Croissant Difference Imaging Pipeline

This folder contains scripts to create, document, and test an AI-ready dataset for transient detection using difference imaging products from the RAPID pipeline with the [Croissant](https://github.com/mlcommons/croissant) metadata format.

## Overview

The pipeline processes astronomical FITS images and catalogs from RAPID job folders, creating a dataset that combines ZOGY and SFFT difference imaging algorithms side-by-side. Each candidate includes 9-channel image tensors (science, reference, ZOGY diff/SCORR, SFFT diff/SCORR, PSF, uncertainties) plus comprehensive photometric features from both algorithms.

### Key Features:
- **Unified Structure**: Single record per candidate with both ZOGY and SFFT measurements
- **Easy Algorithm Comparison**: Side-by-side features enable direct performance analysis  
- **Automatic Truth Generation**: Generates truth tables and cross-matching on-the-fly from source catalogs
- **Injection Tracking**: Identifies artificially injected vs natural transients
- **Truth Linking**: Connects detections to ground truth via cross-matching
- **9-Channel Images**: Complete imaging context including uncertainty maps

## Scripts

### 1. `create_dataset.py`

Processes raw FITS files and catalogs from RAPID job folders and creates a unified normalized dataset with automatic truth table generation.

**What it does:**
- Reads 9 FITS files per job for image channels
- Applies ZScale normalization to each image
- Resizes PSF and uncertainty maps to match other images if necessary
- Stacks them into a 9-channel tensor and saves as `.npy` files
- Copies catalog files for both ZOGY and SFFT pipelines
- **Automatically generates truth tables** from injection catalogs and OpenUniverse catalogs
- **Cross-matches detections** with truth sources (4 pixel radius)
- **Merges ZOGY and SFFT detections** based on detection ID
- Extracts injection status from truth catalogs
- Creates single `candidates.csv` with prefixed ZOGY/SFFT features

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
│   ├── sfftdiffimage_dconv_masked.fits
│   ├── sfftdiffimage_cconv_masked.fits
│   ├── diffpsf.fits
│   ├── diffimage_uncert_masked.fits
│   ├── sfftdiffimage_uncert_masked.fits
│   ├── diffimage_masked_psfcat_finder.txt      # ZOGY finder catalog
│   ├── diffimage_masked_psfcat.txt             # ZOGY PSF photometry
│   ├── sfftdiffimage_masked_psfcat_finder.txt  # SFFT finder catalog
│   ├── sfftdiffimage_masked_psfcat.txt         # SFFT PSF photometry
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
│   ├── jid001.npy         # 9-channel image tensors
│   ├── jid002.npy
│   └── ...
├── catalogs/              # Full catalog files (text only)
│   ├── jid001_diffimage_masked.txt
│   ├── jid001_diffimage_masked_sepsfcat.txt
│   └── ...
└── candidates.csv         # Unified CSV with ZOGY+SFFT features
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

Generates a Croissant metadata file (`croissant.json`) for the unified dataset.

**What it does:**
- Reads the unified `candidates.csv` file
- Creates single record set: `transient_candidates` 
- Includes all ZOGY and SFFT features with proper prefixing
- Documents injection status and truth linking fields
- References 9-channel image tensors and catalog files

**Usage:**
```bash
python generate_croissant.py --dataset_dir <dataset_path> --output <output_json_path>
```

**Arguments:**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--dataset_dir` | `-d` | `./hackathon_dataset` | Dataset directory containing candidates.csv |
| `--output` | `-o` | `./hackathon_dataset/croissant.json` | Output path for `croissant.json` |

---

### 3. `test_croissant_dataset.py`

Comprehensive testing and training script for the unified dataset.

**What it does:**
- Loads unified dataset with single `transient_candidates` record set
- Analyzes dataset composition (injected/natural, filters, detection overlap)
- Reports detection statistics (ZOGY-only, SFFT-only, both)
- Extracts 64x64 cutouts around each candidate position
- Supports flexible feature selection: ZOGY-only, SFFT-only, or combined
- Trains CNN classifiers using 9-channel input with selectable features
- Enables direct algorithm comparison on same candidates

**Usage:**
```bash
# Test loading and analyze dataset composition
python test_croissant_dataset.py --dataset_dir <path_to_dataset>

# Train using only ZOGY features
python test_croissant_dataset.py --dataset_dir <path_to_dataset> --algorithm zogy

# Train using only SFFT features
python test_croissant_dataset.py --dataset_dir <path_to_dataset> --algorithm sfft

# Train using both ZOGY and SFFT features (default)
python test_croissant_dataset.py --dataset_dir <path_to_dataset> --algorithm both
```

**Arguments:**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--dataset_dir` | `-d` | `./hackathon_dataset` | Dataset directory containing `croissant.json` |
| `--algorithm` | `-a` | `both` | Which features to use: `zogy`, `sfft`, or `both` |

---

## Full Pipeline Example

```bash
# Step 1: Create the unified dataset from raw FITS files and catalogs
python create_dataset.py -i ./raw_data -o ./my_dataset

# Step 2: Generate Croissant metadata
python generate_croissant.py -d ./my_dataset -o ./my_dataset/croissant.json

# Step 3: Test dataset and train model with combined features
python test_croissant_dataset.py -d ./my_dataset --algorithm both
```

Install dependencies:
```bash
pip install numpy pandas astropy mlcroissant torch scipy
```

## Dataset Schema

The `candidates.csv` contains unified records with the following columns:

### Core Fields
| Column | Type | Description |
|--------|------|-------------|
| `id` | float | Candidate identifier |
| `x` | float | Pixel x-coordinate |
| `y` | float | Pixel y-coordinate |
| `jid` | string | Job identifier |
| `match_id` | float | Match ID linking to truth table (-1 for no match) |
| `truth_id` | float | ID from combined_truth_table.csv for matched sources |
| `injected` | bool | Whether source was artificially injected |
| `filter` | string | Observation filter band |
| `label` | int | Binary classification (0=bogus, 1=real) |
| `image_filename` | string | Relative path to 9-channel `.npy` tensor |

### ZOGY Features (prefixed with `zogy_`)
| Column | Type | Description |
|--------|------|-------------|
| `zogy_sharpness` | float | Source sharpness metric |
| `zogy_roundness1` | float | First roundness parameter |
| `zogy_roundness2` | float | Second roundness parameter |
| `zogy_npix` | float | Number of pixels above threshold |
| `zogy_peak` | float | Peak pixel intensity |
| `zogy_flux` | float | Measured flux |
| `zogy_mag` | float | Instrumental magnitude |
| `zogy_daofind_mag` | float | DAOFind magnitude |
| `zogy_flags` | float | Quality flags from photometry |

### SFFT Features (prefixed with `sfft_`)
| Column | Type | Description |
|--------|------|-------------|
| `sfft_sharpness` | float | Source sharpness metric |
| `sfft_roundness1` | float | First roundness parameter |
| `sfft_roundness2` | float | Second roundness parameter |
| `sfft_npix` | float | Number of pixels above threshold |
| `sfft_peak` | float | Peak pixel intensity |
| `sfft_flux` | float | Measured flux |
| `sfft_mag` | float | Instrumental magnitude |
| `sfft_daofind_mag` | float | DAOFind magnitude |
| `sfft_flags` | float | Quality flags from photometry |

**Note**: Missing values (NaN) indicate the algorithm did not detect that candidate.

## Image Tensor Format

Each `.npy` file contains a tensor of shape `(H, W, 9)` where the 9 channels are:
1. **Science image** - Background-subtracted science frame
2. **Reference image** - Gain-matched, resampled reference image
3. **ZOGY difference** - ZOGY positive difference image (masked)
4. **ZOGY SCORR** - ZOGY significance map (masked)
5. **SFFT difference** - SFFT decorrelated difference image (masked)
6. **SFFT SCORR** - SFFT cross-convolved significance map (masked)
7. **Difference PSF** - PSF for the difference image
8. **ZOGY uncertainty** - Uncertainty map for ZOGY difference image
9. **SFFT uncertainty** - Uncertainty map for SFFT difference image

All channels are ZScale normalized to the range [0, 1].

## Catalog Products

The `catalogs/` folder contains multiple catalog products per job:

### ZOGY Catalogs:
- `diffimage_masked.txt` - SourceExtractor catalog on ZOGY difference
- `diffimage_masked_sepsfcat.txt` - PSF photometry for SourceExtractor detections
- `diffimage_masked_psfcat_finder.txt` - Photutils DAOStarFinder detection catalog
- `diffimage_masked_psfcat.txt` - PSF photometry for Photutils detections

### SFFT Catalogs:
- `sfftdiffimage_cconv_masked.txt` - SourceExtractor catalog on SFFT cross-convolved image
- `sfftdiffimage_masked_sepsfcat.txt` - PSF photometry for SourceExtractor detections
- `sfftdiffimage_masked_psfcat_finder.txt` - Photutils DAOStarFinder detection catalog
- `sfftdiffimage_masked_psfcat.txt` - PSF photometry for Photutils detections
