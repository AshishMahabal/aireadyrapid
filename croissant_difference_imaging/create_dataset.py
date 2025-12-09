import os
import glob
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval

IMAGE_FILES = {
    "sci": "bkg_subbed_science_image.fits",
    "ref": "awaicgen_output_mosaic_image_resampled_gainmatched.fits",
    "diff_zogy": "diffimage_masked.fits",
    "scorr_zogy": "scorrimage_masked.fits",
    "diff_sfft": "sfftdiffimage_dconv_masked.fits",
    "scorr_sfft": "sfftdiffimage_cconv_masked.fits",
    "psf": "diffpsf.fits"
}

CATALOG_FILES = {
    "zogy_sex": "diffimage_masked.txt",
    "zogy_sepsf": "diffimage_masked_sepsfcat.txt",
    "zogy_finder": "diffimage_masked_psfcat_finder.txt",
    "zogy_psf": "diffimage_masked_psfcat.txt",
    "sfft_sex": "sfftdiffimage_cconv_masked.txt",
    "sfft_sepsf": "sfftdiffimage_masked_sepsfcat.txt",
    "sfft_finder": "sfftdiffimage_masked_psfcat_finder.txt",
    "sfft_psf": "sfftdiffimage_masked_psfcat.txt",
    "diff_unc_zogy": "diffimage_uncert_masked.fits",
    "diff_unc_sfft": "sfftdiffimage_uncert_masked.fits"
}

def load_and_normalize(filepath, target_shape=None):
    """Loads FITS, handles NaNs, and applies ZScale normalization (0-1)"""
    try:
        with fits.open(filepath) as hdu:
            data = hdu[0].data.astype(np.float32)
        data = np.nan_to_num(data)
        
        # Resize if target shape provided (for PSF or mismatched images)
        if target_shape is not None and data.shape != target_shape:
            from scipy.ndimage import zoom
            zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
            data = zoom(data, zoom_factors, order=1)
        
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
        return np.clip((data - vmin) / (vmax - vmin), 0, 1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def process_dataset(input_dir, output_dir):
    images_dir = os.path.join(output_dir, "images")
    catalogs_dir = os.path.join(output_dir, "catalogs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(catalogs_dir, exist_ok=True)
    
    job_folders = sorted(glob.glob(os.path.join(input_dir, "jid*")))
    print(f"Found {len(job_folders)} jobs to process.")
    
    all_zogy_records = []
    all_sfft_records = []

    for job_folder in job_folders:
        job_id = os.path.basename(job_folder)
        print(f"Processing {job_id}...", end=" ")

        channels = []
        valid_image = True
        target_shape = None
        
        for i, key in enumerate(["sci", "ref", "diff_zogy", "scorr_zogy", "diff_sfft", "scorr_sfft", "psf"]):
            path = os.path.join(job_folder, IMAGE_FILES[key])
            if not os.path.exists(path):
                valid_image = False
                break
            
            if i == 0:
                img = load_and_normalize(path)
                if img is None:
                    valid_image = False
                    break
                target_shape = img.shape
                channels.append(img)
            else:
                img = load_and_normalize(path, target_shape=target_shape)
                if img is None:
                    valid_image = False
                    break
                channels.append(img)
        
        if not valid_image:
            print("Skipped (missing/bad images)")
            continue

        full_tensor = np.stack(channels, axis=-1)
        
        npy_filename = f"{job_id}.npy"
        np.save(os.path.join(images_dir, npy_filename), full_tensor)
        print(f"Saved .npy", end=" ")

        for cat_key, cat_file in CATALOG_FILES.items():
            src_path = os.path.join(job_folder, cat_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(catalogs_dir, f"{job_id}_{cat_file}")
                if cat_file.endswith('.fits'):
                    import shutil
                    shutil.copy2(src_path, dst_path)
                else:
                    import shutil
                    shutil.copy2(src_path, dst_path)

        # Extract ZOGY candidates
        zogy_finder_path = os.path.join(job_folder, CATALOG_FILES["zogy_finder"])
        zogy_psf_path = os.path.join(job_folder, CATALOG_FILES["zogy_psf"])
        
        if os.path.exists(zogy_finder_path) and os.path.exists(zogy_psf_path):
            df_finder = pd.read_csv(zogy_finder_path, sep=r'\s+', skipinitialspace=True)
            df_psf = pd.read_csv(zogy_psf_path, sep=r'\s+', skipinitialspace=True)
            
            df_finder = df_finder.rename(columns={'xcentroid': 'x', 'ycentroid': 'y'})
            
            df = df_finder.copy()
            if 'flags' in df_psf.columns:
                df['flags'] = df_psf['flags'].values[:len(df)]
            if 'match' in df_psf.columns:
                df['match'] = df_psf['match'].values[:len(df)]
            
            df['label'] = df['match'].apply(lambda x: 0 if x == -1 else 1)
            df['image_filename'] = f"images/{npy_filename}"
            df['jid'] = job_id
            
            all_zogy_records.append(df)
            print(f"ZOGY: {len(df)} candidates.", end=" ")
        
        # Extract SFFT candidates
        sfft_finder_path = os.path.join(job_folder, CATALOG_FILES["sfft_finder"])
        sfft_psf_path = os.path.join(job_folder, CATALOG_FILES["sfft_psf"])
        
        if os.path.exists(sfft_finder_path) and os.path.exists(sfft_psf_path):
            df_finder = pd.read_csv(sfft_finder_path, sep=r'\s+', skipinitialspace=True)
            df_psf = pd.read_csv(sfft_psf_path, sep=r'\s+', skipinitialspace=True)
            
            df_finder = df_finder.rename(columns={'xcentroid': 'x', 'ycentroid': 'y'})
            
            df = df_finder.copy()
            if 'flags' in df_psf.columns:
                df['flags'] = df_psf['flags'].values[:len(df)]
            if 'match' in df_psf.columns:
                df['match'] = df_psf['match'].values[:len(df)]
            
            df['label'] = df['match'].apply(lambda x: 0 if x == -1 else 1)
            df['image_filename'] = f"images/{npy_filename}"
            df['jid'] = job_id
            
            all_sfft_records.append(df)
            print(f"SFFT: {len(df)} candidates.")
        else:
            print("No SFFT catalogs found.")

    # Save ZOGY candidates
    if all_zogy_records:
        zogy_df = pd.concat(all_zogy_records, ignore_index=True)
        zogy_csv_path = os.path.join(output_dir, "zogy_candidates.csv")
        zogy_df.to_csv(zogy_csv_path, index=False)
        print(f"\nZOGY candidates saved to {zogy_csv_path}")
        print(f"Total ZOGY Candidates: {len(zogy_df)}")
        print(f"ZOGY Real Transients: {zogy_df['label'].sum()}")
    else:
        print("\nNo ZOGY candidates found.")
    
    # Save SFFT candidates
    if all_sfft_records:
        sfft_df = pd.concat(all_sfft_records, ignore_index=True)
        sfft_csv_path = os.path.join(output_dir, "sfft_candidates.csv")
        sfft_df.to_csv(sfft_csv_path, index=False)
        print(f"\nSFFT candidates saved to {sfft_csv_path}")
        print(f"Total SFFT Candidates: {len(sfft_df)}")
        print(f"SFFT Real Transients: {sfft_df['label'].sum()}")
    else:
        print("\nNo SFFT candidates found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create difference imaging dataset from FITS files")
    parser.add_argument("--input_dir", "-i", type=str, default="./mini_dataset",
                        help="Input directory containing job folders (default: ./mini_dataset)")
    parser.add_argument("--output_dir", "-o", type=str, default="./hackathon_dataset",
                        help="Output directory for processed dataset (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
