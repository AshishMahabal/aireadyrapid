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

def process_dataset(input_dir, output_dir):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    job_folders = sorted(glob.glob(os.path.join(input_dir, "jid*")))
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
        np.save(os.path.join(images_dir, npy_filename), full_tensor)
        print(f"Saved .npy", end=" ")

        cat_path = os.path.join(job_folder, "psfcat.csv")
        finder_path = os.path.join(job_folder, "diffimage_masked_psfcat_finder.txt")
        
        if os.path.exists(cat_path) and os.path.exists(finder_path):
            df_psfcat = pd.read_csv(cat_path)
            df_finder = pd.read_csv(finder_path, sep=r'\s+', skipinitialspace=True)
            df_finder = df_finder.rename(columns={'xcentroid': 'x', 'ycentroid': 'y'})
            
            df = df_finder.copy()
            if 'flags' in df_psfcat.columns:
                df['flags'] = df_psfcat['flags']
            if 'match' in df_psfcat.columns:
                df['match'] = df_psfcat['match']
            
            df['label'] = df['match'].apply(lambda x: 0 if x == -1 else 1)
            df['image_filename'] = f"images/{npy_filename}"
            df['jid'] = job_id
            
            all_records.append(df)
            print(f"Extracted {len(df)} candidates.")
        else:
            print("No catalog found.")

    if all_records:
        master_df = pd.concat(all_records)
        master_csv_path = os.path.join(output_dir, "master_index.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nDone! Master Index saved to {master_csv_path}")
        print(f"Total Candidates: {len(master_df)}")
        print(f"Real Transients: {master_df['label'].sum()}")
    else:
        print("\nFailed to create dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset from FITS files")
    parser.add_argument("--input_dir", "-i", type=str, default="./mini_dataset",
                        help="Input directory containing job folders (default: ./mini_dataset)")
    parser.add_argument("--output_dir", "-o", type=str, default="./hackathon_dataset",
                        help="Output directory for processed dataset (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)