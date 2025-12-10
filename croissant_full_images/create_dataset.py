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
INJECTED_CATALOGS = r'Roman_TDS_simple_model_([a-zA-Z])(\d+)_(\d+)_(\d+)_lite_inject\.txt'
OPENUNIVERSE_CATALOGS = r'Roman_TDS_index_([a-zA-Z])(\d+)_(\d+)_(\d+)\.txt'
PSF_CATALOGS = r'diffimage_masked_psfcat.txt'
FEATS_CATALOGS = r'diffimage_masked_psfcat_finder.txt'
MAG_LIM = 26.0
CROSS_MATCH_MAX_SEP = 4.0  # pixels


def load_field_list(field_files, selected_fields=None, mjd_upper_bound=None):
    """
    Load field list from one or more text files and filter by fields and MJD.
    
    Args:
        field_files: List of paths to field list text files
        selected_fields: List of field IDs to include (None = all fields)
        mjd_upper_bound: Maximum MJD value (None = no limit)
    
    Returns:
        List of job IDs that match the criteria
    """
    all_data = []
    
    for field_file in field_files:
        df = pd.read_csv(field_file, delim_whitespace=True)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Apply field filter
    if selected_fields is not None:
        combined_df = combined_df[combined_df['FIELD'].isin(selected_fields)]
    
    # Apply MJD upper bound
    if mjd_upper_bound is not None:
        combined_df = combined_df[combined_df['MJD'] <= mjd_upper_bound]
    
    # Extract job IDs
    job_ids = combined_df['JOB'].unique().tolist()
    
    return sorted(job_ids)

FILES = {
    "sci": "bkg_subbed_science_image.fits",
    "ref": "awaicgen_output_mosaic_image_resampled_gainmatched.fits",
    "diff": "diffimage_masked.fits",
    "scorr": "scorrimage_masked.fits"
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
                filter_val = fits.getheader(os.path.join(root, file))['FILTER']
                concat_truth_tables['filter'] = filter_val
                concat_truth_tables['zpt'] = zpt
    
    # Load PSF catalogs and match
    psf_feats_path = os.path.join(jid_folder, FEATS_CATALOGS)
    psf_cats_path = os.path.join(jid_folder, PSF_CATALOGS)
    
    if not os.path.exists(psf_feats_path) or not os.path.exists(psf_cats_path):
        return concat_truth_tables, pd.DataFrame()
    
    psf_feats = pd.DataFrame(np.genfromtxt(psf_feats_path, dtype=None, names=True, encoding="utf-8"))
    psf_cats = load_truth_file(psf_cats_path).to_pandas()
    psf_cats = psf_cats[['id', 'x_fit', 'y_fit']]
    psf_cats = psf_cats.rename(columns={'x_fit': 'x', 'y_fit': 'y'})
    psf_cats = psf_cats.join(psf_feats.set_index('id'), on='id', how='left', rsuffix='_feat')
    psf_cats = crossmatch_detections(concat_truth_tables, psf_cats, max_sep=CROSS_MATCH_MAX_SEP)
    psf_cats = psf_cats[(0 <= psf_cats['x']) & (psf_cats['x'] < 4090) & (0 <= psf_cats['y']) & (psf_cats['y'] < 4090)]
    
    return concat_truth_tables, psf_cats

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

def process_dataset(input_dir, output_dir, field_files=None, selected_fields=None, mjd_upper_bound=None):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load job IDs from field list files
    if field_files:
        job_ids = load_field_list(field_files, selected_fields, mjd_upper_bound)
        print(f"Loaded {len(job_ids)} jobs from field lists.")
        if selected_fields:
            print(f"Filtered by fields: {selected_fields}")
        if mjd_upper_bound:
            print(f"MJD upper bound: {mjd_upper_bound}")
    else:
        # Fallback: discover all jid* folders
        job_ids = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(input_dir, "jid*")))]
        print(f"No field list provided. Discovered {len(job_ids)} jobs.")
    
    job_folders = [os.path.join(input_dir, jid) for jid in job_ids if os.path.exists(os.path.join(input_dir, jid))]
    print(f"Processing {len(job_folders)} jobs.")
    
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

        # Generate truth table and matched catalogs on the fly
        truth_df, psf_match_df = process_detections(job_folder)
        
        if psf_match_df.empty:
            print("No detections found.")
            continue
        
        df = psf_match_df.copy()
        
        # Link to truth table via match_id (exactly as in cutouts)
        if truth_df is not None and len(truth_df) > 0 and 'match_id' in df.columns:
            # Clean match_id values - remove extra whitespace and handle empty strings
            df['match_id'] = df['match_id'].astype(str).str.strip()
            df['match_id'] = df['match_id'].replace('', pd.NA)
            df['match_id'] = df['match_id'].replace('nan', pd.NA)
            
            # match_id directly corresponds to truth table 'id' column
            df['truth_id'] = df['match_id']
            
            # Merge with truth table using truth_id
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
            df = pd.merge(
                df,
                truth_merge,
                on='truth_id',
                how='left'
            )
            
            # Fill missing injection status with False
            if 'injected' in df.columns:
                df['injected'] = df['injected'].fillna(False).astype(bool)
        else:
            # No truth table - add default columns
            df['truth_id'] = pd.NA
            df['injected'] = False
        
        # Set label: real (1) if match_id is present, bogus (0) if match_id is missing/empty
        def compute_label(match_id):
            if pd.isna(match_id):
                return 0
            match_str = str(match_id).strip()
            if match_str == '' or match_str == 'nan' or match_str == '-1' or match_str == 'None':
                return 0
            return 1
        
        df['label'] = df['match_id'].apply(compute_label)
        
        df['image_filename'] = f"images/{npy_filename}"
        df['jid'] = job_id
        
        all_records.append(df)
        print(f"Extracted {len(df)} candidates.")

    if all_records:
        master_df = pd.concat(all_records, ignore_index=True)
        
        # Replace empty strings with NA for proper CSV handling
        master_df['match_id'] = master_df['match_id'].replace('', pd.NA)
        master_df['truth_id'] = master_df['truth_id'].replace('', pd.NA)
        
        master_csv_path = os.path.join(output_dir, "master_index.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nDone! Master Index saved to {master_csv_path}")
        print(f"Total Candidates: {len(master_df)}")
        print(f"Real Transients: {master_df['label'].sum()}")
        print(f"Bogus Detections: {(master_df['label'] == 0).sum()}")
        if 'injected' in master_df.columns:
            print(f"Injected Sources: {master_df['injected'].sum()}")
            print(f"Natural Sources: {(master_df['injected'] == False).sum()}")
    else:
        print("\nFailed to create dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Croissant full images dataset")
    parser.add_argument("--input_dir", "-i", type=str, default="./mini_dataset",
                        help="Input directory containing job folders (default: ./mini_dataset)")
    parser.add_argument("--output_dir", "-o", type=str, default="./hackathon_dataset",
                        help="Output directory for processed dataset (default: ./hackathon_dataset)")
    parser.add_argument("--field_files", nargs="+", help="Field list text files (e.g., H158_fields.txt R062_fields.txt)")
    parser.add_argument("--fields", nargs="+", type=int, help="Specific field IDs to process")
    parser.add_argument("--mjd_max", type=float, help="Maximum MJD value (upper bound)")
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.field_files, args.fields, args.mjd_max)
