import os
import glob
import argparse
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval

# Regex patterns for file matching
SCI_IMAGE = r'Roman_TDS_simple_model_([a-zA-Z])(\d+)_(\d+)_(\d+)_lite_inject_reformatted\.fits'
REF_IMAGE = r'awaicgen_output_mosaic_image_resampled_gainmatched.fits'
DIFF_IMAGE = r'diffimage_masked.fits'
INJECTED_CATALOGS = r'Roman_TDS_simple_model_([a-zA-Z])(\d+)_(\d+)_(\d+)_lite_inject\.txt'
OPENUNIVERSE_CATALOGS = r'Roman_TDS_index_([a-zA-Z])(\d+)_(\d+)_(\d+)\.txt'
PSF_CATALOGS = r'diffimage_masked_psfcat.txt'
FEATS_CATALOGS = r'diffimage_masked_psfcat_finder.txt'
MAG_LIM = 26.0
CROSS_MATCH_MAX_SEP = 4.0  # pixels

IMAGE_FILES = {
    "sci": "bkg_subbed_science_image.fits",
    "ref": "awaicgen_output_mosaic_image_resampled_gainmatched.fits",
    "diff_zogy": "diffimage_masked.fits",
    "scorr_zogy": "scorrimage_masked.fits",
    "diff_sfft": "sfftdiffimage_dconv_masked.fits",
    "scorr_sfft": "sfftdiffimage_cconv_masked.fits",
    "psf": "diffpsf.fits",
    "unc_zogy": "diffimage_uncert_masked.fits",
    "unc_sfft": "sfftdiffimage_uncert_masked.fits"
}

CATALOG_FILES = {
    "zogy_sex": "diffimage_masked.txt",
    "zogy_sepsf": "diffimage_masked_sepsfcat.txt",
    "zogy_finder": "diffimage_masked_psfcat_finder.txt",
    "zogy_psf": "diffimage_masked_psfcat.txt",
    "sfft_sex": "sfftdiffimage_cconv_masked.txt",
    "sfft_sepsf": "sfftdiffimage_masked_sepsfcat.txt",
    "sfft_finder": "sfftdiffimage_masked_psfcat_finder.txt",
    "sfft_psf": "sfftdiffimage_masked_psfcat.txt"
}

def load_truth_file(filepath):
    """Use astropy to read truth file (whitespace separated, header present)"""
    try:
        table = Table.read(filepath, format="ascii")
        return table
    except Exception as e:
        print(f"Error loading truth file {filepath}: {e}")
        return Table()

def crossmatch_detections(truth_table, detected_table, max_sep=4.0):
    """Cross-match detected sources with truth table"""
    detected_table['match_id'] = ''
    
    for _, truth_row in truth_table.iterrows():
        if truth_row['mag'] + truth_row['zpt'] > MAG_LIM:
            continue  # Skip faint sources
        x_truth = truth_row['x']
        y_truth = truth_row['y']
        
        # Calculate distances to all detected sources
        dx = detected_table['x'] - x_truth
        dy = detected_table['y'] - y_truth
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find the closest detected source within max_sep
        min_distance = distances.min()
        if min_distance <= max_sep:
            match_idx = distances.idxmin()
            detected_table.at[match_idx, 'match_id'] = truth_row['id']
    
    return detected_table

def process_detections(jid_folder):
    """Process detections and generate truth table and matched catalogs on the fly"""
    concat_truth_tables = pd.DataFrame(columns=['id', 'x', 'y', 'ra', 'dec', 'mag', 'zpt', 'realized_flux', 'flux', 'filter', 'injected_image', 'jid_folder'])
    
    for root, dirs, files in os.walk(jid_folder):
        for file in files:
            if re.match(INJECTED_CATALOGS, file):
                catalog_path = os.path.join(root, file)
                truth_table = load_truth_file(catalog_path).to_pandas()
                truth_table['injected_image'] = True
                truth_table['jid_folder'] = jid_folder
                truth_table = truth_table.rename(columns={'xpix': 'x', 'ypix': 'y'})
                truth_table['id'] = truth_table.index.map(lambda i: f"{i}_inj")
                concat_truth_tables = pd.concat([concat_truth_tables, truth_table], ignore_index=True)
            elif re.match(OPENUNIVERSE_CATALOGS, file):
                catalog_path = os.path.join(root, file)
                truth_table = load_truth_file(catalog_path).to_pandas()
                truth_table = truth_table[truth_table['obj_type'] == 'transient']
                truth_table['injected_image'] = False
                truth_table['jid_folder'] = jid_folder
                truth_table.drop(columns=['obj_type'], inplace=True)
                truth_table = truth_table.rename(columns={'object_id': 'id'})
                truth_table['id'] = truth_table['id'].map(lambda i: f"{i}_ou")
                concat_truth_tables = pd.concat([concat_truth_tables, truth_table], ignore_index=True)
            elif re.match(SCI_IMAGE, file):
                zpt = fits.getheader(os.path.join(root, file))['ZPTMAG']
                filter = fits.getheader(os.path.join(root, file))['FILTER']
                concat_truth_tables['filter'] = filter
                concat_truth_tables['zpt'] = zpt
    
    # Load PSF catalogs and match
    psf_feats = pd.DataFrame(np.genfromtxt(os.path.join(jid_folder, FEATS_CATALOGS), dtype=None, names=True, encoding="utf-8"))
    psf_cats = load_truth_file(os.path.join(jid_folder, PSF_CATALOGS)).to_pandas()
    psf_cats = psf_cats[['id', 'x_fit', 'y_fit']]
    psf_cats = psf_cats.rename(columns={'x_fit': 'x', 'y_fit': 'y'})
    psf_cats = psf_cats.join(psf_feats.set_index('id'), on='id', how='left', rsuffix='_feat')
    psf_cats = crossmatch_detections(concat_truth_tables, psf_cats, max_sep=CROSS_MATCH_MAX_SEP)
    psf_cats = psf_cats[(0 <= psf_cats['x']) & (psf_cats['x'] < 4090) & (0 <= psf_cats['y']) & (psf_cats['y'] < 4090)]
    
    return concat_truth_tables, psf_cats

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
    
    all_records = []

    for job_folder in job_folders:
        job_id = os.path.basename(job_folder)
        print(f"Processing {job_id}...", end=" ")

        channels = []
        valid_image = True
        target_shape = None
        
        for i, key in enumerate(["sci", "ref", "diff_zogy", "scorr_zogy", "diff_sfft", "scorr_sfft", "psf", "unc_zogy", "unc_sfft"]):
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
                import shutil
                shutil.copy2(src_path, dst_path)

        # Generate truth table and matched catalogs on the fly
        print(f"Processing detections...", end=" ")
        truth_df, psf_match_df = process_detections(job_folder)
        
        # Extract ZOGY candidates from psf_cats_match (already has match_id)
        zogy_df = None
        if psf_match_df is not None and len(psf_match_df) > 0:
            # Use psf_cats_match as the source for ZOGY data
            zogy_df = psf_match_df.copy()
            
            # Rename columns for ZOGY
            zogy_cols_to_rename = {
                'sharpness': 'zogy_sharpness',
                'roundness1': 'zogy_roundness1',
                'roundness2': 'zogy_roundness2',
                'npix': 'zogy_npix',
                'peak': 'zogy_peak',
                'flux': 'zogy_flux',
                'mag': 'zogy_mag',
                'daofind_mag': 'zogy_daofind_mag'
            }
            for old_col, new_col in zogy_cols_to_rename.items():
                if old_col in zogy_df.columns:
                    zogy_df.rename(columns={old_col: new_col}, inplace=True)
            
            # Add ZOGY flags (set to 0 if not present)
            zogy_df['zogy_flags'] = 0.0
            
            print(f"ZOGY: {len(zogy_df)} candidates.", end=" ")
        
        # Extract SFFT candidates
        sfft_finder_path = os.path.join(job_folder, CATALOG_FILES["sfft_finder"])
        sfft_psf_path = os.path.join(job_folder, CATALOG_FILES["sfft_psf"])
        
        sfft_df = None
        if os.path.exists(sfft_finder_path) and os.path.exists(sfft_psf_path):
            df_finder = pd.read_csv(sfft_finder_path, sep=r'\s+', skipinitialspace=True)
            df_psf = pd.read_csv(sfft_psf_path, sep=r'\s+', skipinitialspace=True)
            
            # Clean column names
            df_finder.columns = df_finder.columns.str.strip()
            df_psf.columns = df_psf.columns.str.strip()
            
            df_finder = df_finder.rename(columns={'xcentroid': 'x', 'ycentroid': 'y'})
            
            sfft_df = df_finder.copy()
            
            # Try to match SFFT to ZOGY by id to get match_id
            if zogy_df is not None and 'id' in sfft_df.columns and 'id' in zogy_df.columns:
                # Merge to get match_id from ZOGY data
                sfft_df = pd.merge(
                    sfft_df,
                    zogy_df[['id', 'match_id']],
                    on='id',
                    how='left'
                )
            
            if 'flags' in df_psf.columns:
                sfft_df['sfft_flags'] = df_psf['flags'].values[:len(sfft_df)]
            else:
                sfft_df['sfft_flags'] = 0.0
            
            # Rename SFFT-specific columns
            sfft_cols_to_rename = {
                'sharpness': 'sfft_sharpness',
                'roundness1': 'sfft_roundness1',
                'roundness2': 'sfft_roundness2',
                'npix': 'sfft_npix',
                'peak': 'sfft_peak',
                'flux': 'sfft_flux',
                'mag': 'sfft_mag',
                'daofind_mag': 'sfft_daofind_mag'
            }
            for old_col, new_col in sfft_cols_to_rename.items():
                if old_col in sfft_df.columns:
                    sfft_df.rename(columns={old_col: new_col}, inplace=True)
            
            print(f"SFFT: {len(sfft_df)} candidates.", end=" ")
        
        # Merge ZOGY and SFFT based on detection id
        if zogy_df is not None and sfft_df is not None:
            # Merge on 'id' - the unique detection identifier
            # This links the same detection from ZOGY and SFFT algorithms
            merged_df = pd.merge(
                zogy_df, 
                sfft_df, 
                on='id', 
                how='outer',
                suffixes=('_zogy', '_sfft')
            )
            
            # Use ZOGY coordinates if available, otherwise SFFT
            if 'x_zogy' in merged_df.columns and 'x_sfft' in merged_df.columns:
                merged_df['x'] = merged_df['x_zogy'].fillna(merged_df['x_sfft'])
                merged_df['y'] = merged_df['y_zogy'].fillna(merged_df['y_sfft'])
                merged_df.drop(columns=['x_zogy', 'y_zogy', 'x_sfft', 'y_sfft'], inplace=True)
            
            # Use ZOGY match_id if available, otherwise SFFT match_id
            if 'match_id_zogy' in merged_df.columns and 'match_id_sfft' in merged_df.columns:
                merged_df['match_id'] = merged_df['match_id_zogy'].fillna(merged_df['match_id_sfft'])
                merged_df.drop(columns=['match_id_zogy', 'match_id_sfft'], inplace=True)
            
        elif zogy_df is not None:
            merged_df = zogy_df
        elif sfft_df is not None:
            merged_df = sfft_df
        else:
            print("No candidates found.")
            continue
        
        # Link to truth table via match_id
        if truth_df is not None and len(truth_df) > 0 and 'match_id' in merged_df.columns:
            # Clean match_id values - remove extra whitespace and handle empty strings
            merged_df['match_id'] = merged_df['match_id'].astype(str).str.strip()
            merged_df['match_id'] = merged_df['match_id'].replace('', pd.NA)
            merged_df['match_id'] = merged_df['match_id'].replace('nan', pd.NA)
            
            # match_id directly corresponds to truth table 'id' column
            # Extract base truth_id from match_id (e.g., "18_inj" or just "18")
            merged_df['truth_id'] = merged_df['match_id']
            
            # Merge with truth table using truth_id
            # Rename truth_df columns for clarity
            truth_merge = truth_df[['id', 'injected_image']].copy()
            truth_merge.columns = ['truth_id', 'injected']
            truth_merge['truth_id'] = truth_merge['truth_id'].astype(str).str.strip()
            
            # Convert injected_image to boolean (handle string "True"/"False")
            def parse_bool(val):
                if pd.isna(val):
                    return False
                if isinstance(val, bool):
                    return val
                val_str = str(val).strip().lower()
                return val_str in ['true', '1', 'yes']
            
            truth_merge['injected'] = truth_merge['injected'].apply(parse_bool)
            
            # Merge
            merged_df = pd.merge(
                merged_df,
                truth_merge,
                on='truth_id',
                how='left'
            )
            
            # Fill missing injection status with False (real/bogus transients not in truth table)
            if 'injected' in merged_df.columns:
                merged_df['injected'] = merged_df['injected'].fillna(False).astype(bool)
            
            num_injected = int(merged_df['injected'].sum())
            print(f"Linked to truth: {num_injected} injected sources.")
        else:
            # No truth table - add default columns
            merged_df['truth_id'] = pd.NA
            merged_df['injected'] = False
        
        # Set label: real (1) if match_id is present, bogus (0) if match_id is missing/empty
        def compute_label(match_id):
            if pd.isna(match_id):
                return 0
            match_str = str(match_id).strip()
            if match_str == '' or match_str == 'nan' or match_str == '-1' or match_str == 'None':
                return 0
            return 1
        
        merged_df['label'] = merged_df['match_id'].apply(compute_label)
        
        # Add metadata
        merged_df['image_filename'] = f"images/{npy_filename}"
        merged_df['jid'] = job_id
        
        all_records.append(merged_df)
        print(f"Merged: {len(merged_df)} total candidates.")

    # Save unified candidates
    if all_records:
        candidates_df = pd.concat(all_records, ignore_index=True)
        
        # Replace empty strings with NA for proper CSV handling
        candidates_df['match_id'] = candidates_df['match_id'].replace('', pd.NA)
        candidates_df['truth_id'] = candidates_df['truth_id'].replace('', pd.NA)
        
        candidates_csv_path = os.path.join(output_dir, "candidates.csv")
        candidates_df.to_csv(candidates_csv_path, index=False)
        print(f"\nCandidates saved to {candidates_csv_path}")
        print(f"Total Candidates: {len(candidates_df)}")
        print(f"Real Transients: {candidates_df['label'].sum()}")
        print(f"Bogus Detections: {(candidates_df['label'] == 0).sum()}")
        if 'injected' in candidates_df.columns:
            print(f"Injected Sources: {candidates_df['injected'].sum()}")
            print(f"Natural Sources: {(candidates_df['injected'] == False).sum()}")
    else:
        print("\nNo candidates found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create difference imaging dataset from FITS files")
    parser.add_argument("--input_dir", "-i", type=str, default="./mini_dataset",
                        help="Input directory containing job folders (default: ./mini_dataset)")
    parser.add_argument("--output_dir", "-o", type=str, default="./hackathon_dataset",
                        help="Output directory for processed dataset (default: ./hackathon_dataset)")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir)
