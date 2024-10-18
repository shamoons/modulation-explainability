# src/convert_to_constellation.py
import logging
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import get_dataloader
from utils.constellation_data_processing_utils import process_samples, group_by_modulation_snr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    """
    Parse command-line arguments for SNR, modulation types, and limit.
    """
    parser = argparse.ArgumentParser(description='Convert IQ data to constellation images.')

    # Add arguments for limit, snr_list, mod_list, batch_size, image_types, and workers
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process.')
    parser.add_argument('--snr_list', type=str, default=None, help='Comma-separated list of SNRs to process.')
    parser.add_argument('--mod_list', type=str, default=None, help='Comma-separated list of modulation types to process.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for processing.')
    parser.add_argument('--image_types', type=str, default='grayscale', help='Comma-separated list of image types to generate.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for parallel processing.')

    return parser.parse_args()


def process_modulation_snr_set(modulation_type, snr, samples, batch_size, image_types):
    """
    Function to process a single modulation/SNR set.
    This function will be executed by each worker.
    """
    total_samples = len(samples)

    tqdm_desc = f'Processing {modulation_type} at SNR {snr}'
    for batch_start_idx in tqdm(range(0, total_samples, batch_size), desc=tqdm_desc):
        batch_samples = samples[batch_start_idx:batch_start_idx + batch_size]
        process_samples(
            np.array(batch_samples), modulation_type, snr,
            batch_start_idx, 'constellation', (224, 224), image_types
        )


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Convert comma-separated SNR list to a list of integers
    if args.snr_list is not None:
        snrs_to_process = [int(snr) for snr in args.snr_list.split(',')]
    else:
        snrs_to_process = None

    # Convert comma-separated modulation list to a list of strings
    if args.mod_list is not None:
        mods_to_process = args.mod_list.split(',')
    else:
        mods_to_process = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']

    # Set a limit for the number of samples to process, default is None (no limit)
    limit = args.limit

    # Define image types, including "raw"
    image_types = args.image_types.split(',')
    print(f"Generating images for the following types: {image_types}")

    # Get the dataloader and process the data
    dataloader, mod2int = get_dataloader(batch_size=args.batch_size, snr_list=snrs_to_process, mods_to_process=mods_to_process, limit=limit)

    # Group data based on specified SNRs and modulations
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Use ThreadPoolExecutor to process each modulation/SNR set in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        # For each modulation/SNR set, submit a separate task to the executor
        for modulation_type, snr_dict in grouped_data.items():
            for snr, samples in snr_dict.items():
                futures.append(
                    executor.submit(process_modulation_snr_set, modulation_type, snr, samples, 32, image_types)
                )

        # Use as_completed to process results as they complete
        for future in as_completed(futures):
            try:
                future.result()  # This will raise any exceptions caught during processing
            except Exception as e:
                logging.error(f"Error in worker: {e}")
