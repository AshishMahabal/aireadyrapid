import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval

INPUT_DIR = "./mini_dataset"
OUTPUT_DIR = "./hackathon_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

FILES = {
    "sci": "bkg_subbed_science_image.fits",
    "ref": "awaicgen_output_mosaic_image_resampled.fits",
    "diff": "diffimage_masked.fits",
    "scorr": "scorrimage_masked.fits"
}

def load_and_normalize(filepath):
    """Loads FITS, handles NaNs, and apply ZScale normalization (0-1)"""
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

def process_dataset():
    job_folders = sorted(glob.glob(os.path.join(INPUT_DIR, "jid*")))
    print(f"Found {len(job_folders)} jobs to process.")
    
    all_records = []

    for job_folder in job_folders:
        job_id = os.path.basename(job_folder)
        print(f"Processing {job_id}...", end=" ")

        channels = []
        valid_image = True
        for key in ["sci", "ref", "diff", "scorr"]:
            path = os.path.join(job_folder, FILES[key])
            if not os.path.exists(path):
                valid_image = False; break
            img = load_and_normalize(path)
            if img is None:
                valid_image = False; break
            channels.append(img)
        
        if not valid_image:
            print("Skipped (missing/bad images)")
            continue

        full_tensor = np.stack(channels, axis=-1)
        
        npy_filename = f"{job_id}.npy"
        np.save(os.path.join(IMAGES_DIR, npy_filename), full_tensor)
        print(f"Saved .npy", end=" ")

        cat_path = os.path.join(job_folder, "psfcat.csv")
        if os.path.exists(cat_path):
            df = pd.read_csv(cat_path)
            
            # Convert 'match' column to a binary label (0=bogus, 1=real)
            # Assumes -1 is bogus, anything else is real
            df['label'] = df['match'].apply(lambda x: 0 if x == -1 else 1)
            
            df['image_filename'] = f"images/{npy_filename}"
            
            cols_to_keep = ['x', 'y', 'flux', 'fwhm', 'elongation', 'label', 'image_filename']

            existing_cols = [c for c in cols_to_keep if c in df.columns]
            df_clean = df[existing_cols].copy()
            df_clean['job_id'] = job_id
            
            all_records.append(df_clean)
            print(f"Extracted {len(df)} candidates.")
        else:
            print("No catalog found.")

    if all_records:
        master_df = pd.concat(all_records)
        master_csv_path = os.path.join(OUTPUT_DIR, "master_index.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nDone! Master Index saved to {master_csv_path}")
        print(f"Total Candidates: {len(master_df)}")
        print(f"Real Transients: {master_df['label'].sum()}")
    else:
        print("\nFailed to create dataset.")

if __name__ == "__main__":
    process_dataset()