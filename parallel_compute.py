import os
import sys
import dill
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from pyspi.calculator import Calculator
from pyspi.data import Data
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pyspi_compute_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Absolute paths
NPY_DIR = Path('/media/dan/Data/git/ubiquitous-spork/bipolar_reref/random_three_pid/run_data/npy')
OUTPUT_DIR = Path('/media/dan/Data/git/ubiquitous-spork/bipolar_reref/random_three_pid/run_data/outputs')
CONFIG_FILE = Path('/media/dan/Data/git/ubiquitous-spork/bipolar_reref/random_three_pid/selected_config.yaml')

def process_file(npy_file):
    """Process a single .npy file and save results."""
    try:
        # Get the base name without extension
        base_name = Path(npy_file).stem
        
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Output file path
        output_file = OUTPUT_DIR / f"{base_name}.pkl"
        
        # Skip if output already exists
        if output_file.exists():
            logging.info(f"Skipping {base_name} - output already exists")
            return True
            
        # Load and process data
        data = Data(data=str(npy_file), dim_order='ps', name=base_name, normalise=True, detrend=True)
        
        # Create calculator with config file
        calc = Calculator(configfile=str(CONFIG_FILE))
        calc.load_dataset(data)
        calc.compute()
        print('done')
        
        # Save only the table
        with open(output_file, 'wb') as f:
            dill.dump(calc.table, f)
        
        logging.info(f"Successfully processed {base_name}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {npy_file}: {str(e)}")
        return False

def main():
    # Verify config file exists
    if not CONFIG_FILE.exists():
        logging.error(f"Config file {CONFIG_FILE} does not exist")
        sys.exit(1)

    # Get list of .npy files
    if not NPY_DIR.exists():
        logging.error(f"Directory {NPY_DIR} does not exist")
        sys.exit(1)
        
    npy_files = list(NPY_DIR.glob('*.npy'))
    if not npy_files:
        logging.error(f"No .npy files found in {NPY_DIR}")
        sys.exit(1)
    
    # Take only first 4 files for testing
    # npy_files = npy_files[:4]
    total_files = len(npy_files)
    # logging.info(f"Testing with {total_files} files")
    
    # Process files in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, npy_files), total=total_files))

    # for file in npy_files:
    #     process_file(file)
    
    # Report results
    successful = sum(results)
    failed = total_files - successful
    logging.info(f"Processing complete. Successful: {successful}, Failed: {failed}")

if __name__ == '__main__':
    main() 
