import os
import glob
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval

FILES = {
    "sci": "bkg_subbed_science_image.fits",
    "ref": "awaicgen_output_mosaic_image_resampled.fits",
    "diff": "diffimage_masked.fits",
    "scorr": "scorrimage_masked.fits"
}

CUTOUT_SIZE = 64


def load_and_normalize(filepath):
    """Loads FITS file, handles NaNs, and applies ZScale normalization (0-1)."""
    try:
        with fits.open(filepath) as hdu:
            data = hdu[0].data.astype(np.float32)
        data = np.nan_to_num(data)
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def extract_cutout(image, x, y, size=64):
    half = size // 2
    h, w = image.shape
    x, y = int(x), int(y)

    y_min = max(0, y - half)
    y_max = min(h, y + half)
    x_min = max(0, x - half)
    x_max = min(w, x + half)

    cutout = image[y_min:y_max, x_min:x_max]

    pad_h = size - cutout.shape[0]
    pad_w = size - cutout.shape[1]
    if pad_h > 0 or pad_w > 0:
        cutout = np.pad(cutout, ((0, pad_h), (0, pad_w)), mode='constant')

    return cutout


def process_dataset(input_dir, output_dir):
    cutouts_dir = os.path.join(output_dir, "cutouts")
    os.makedirs(cutouts_dir, exist_ok=True)
    
    job_folders = sorted(glob.glob(os.path.join(input_dir, "jid*")))
    print(f"Found {len(job_folders)} jobs to process.")
    
    all_records = []
    cutout_counter = 0

    for job_folder in job_folders:
        job_id = os.path.basename(job_folder)
        print(f"Processing {job_id}...", end=" ")

        # Load and normalize all 4 images
        images = {}
        valid_job = True
        for key in ["sci", "ref", "diff", "scorr"]:
            path = os.path.join(job_folder, FILES[key])
            if not os.path.exists(path):
                valid_job = False
                break
            img = load_and_normalize(path)
            if img is None:
                valid_job = False
                break
            images[key] = img
        
        if not valid_job:
            print("Skipped (missing/bad images)")
            continue

        # Load catalog
        cat_path = os.path.join(job_folder, "psfcat.csv")
        if not os.path.exists(cat_path):
            print("No catalog found.")
            continue
            
        df = pd.read_csv(cat_path)
        job_cutout_count = 0
        
        for idx, row in df.iterrows():
            x, y = row['x'], row['y']
            
            channels = []
            for key in ["sci", "ref", "diff", "scorr"]:
                cutout = extract_cutout(images[key], x, y, CUTOUT_SIZE)
                channels.append(cutout)
            
            # Stack into 4-channel tensor (64, 64, 4)
            tensor = np.stack(channels, axis=-1)
            
            # Save cutout
            cutout_filename = f"{job_id}_cutout{job_cutout_count:05d}.npy"
            np.save(os.path.join(cutouts_dir, cutout_filename), tensor)
            
            # Convert 'match' column to a binary label (0=bogus, 1=real)
            label = 0 if row['match'] == -1 else 1
            
            record = {
                'cutout_id': cutout_counter,
                'job_id': job_id,
                'x': x,
                'y': y,
                'label': label,
                'cutout_filename': f"cutouts/{cutout_filename}"
            }
            
            # Add optional columns if they exist
            for col in ['flux', 'fwhm', 'elongation']:
                if col in row:
                    record[col] = row[col]
            
            all_records.append(record)
            cutout_counter += 1
            job_cutout_count += 1
        
        print(f"Extracted {job_cutout_count} cutouts.")

    if all_records:
        master_df = pd.DataFrame(all_records)
        master_csv_path = os.path.join(output_dir, "master_index.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nDone! Master Index saved to {master_csv_path}")
        print(f"Total Cutouts: {len(master_df)}")
        print(f"Real Transients: {master_df['label'].sum()}")
    else:
        print("\nFailed to create dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create cutout dataset from FITS files")
    parser.add_argument("--input_dir", "-i", type=str, default="./mini_dataset",
                        help="Input directory containing job folders (default: ./mini_dataset)")
    parser.add_argument("--output_dir", "-o", type=str, default="./hackathon_dataset",
                        help="Output directory for processed dataset (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
