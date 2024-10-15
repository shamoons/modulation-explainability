# src/convert_to_constellation.py
import logging
import argparse
from tqdm import tqdm
from data_loader import get_dataloader
from utils.constellation_data_processing_utils import process_sample, group_by_modulation_snr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    """
    Parse command-line arguments for SNR, modulation types, and limit.
    """
    parser = argparse.ArgumentParser(description='Convert IQ data to constellation images.')

    # Add arguments for limit, snr_list, and mod_list
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to process.')
    parser.add_argument('--snr_list', type=str, default=None, help='Comma-separated list of SNRs to process.')
    parser.add_argument('--mod_list', type=str, default=None, help='Comma-separated list of modulation types to process.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for processing.')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Convert comma-separated SNR list to a list of integers
    if args.snr_list is not None:
        snrs_to_process = [int(snr) for snr in args.snr_list.split(',')]
    else:
        snrs_to_process = [20, 22, 24, 26, 28, 30]  # Default SNRs if not provided

    # Convert comma-separated modulation list to a list of strings
    if args.mod_list is not None:
        mods_to_process = args.mod_list.split(',')
    else:
        mods_to_process = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']  # Default modulations if not provided

    # Set a limit for the number of samples to process, default is None (no limit)
    limit = args.limit

    # Define image types, including "raw"
    image_types = ['grayscale']

    # Get the dataloader and process the data
    dataloader, mod2int = get_dataloader(batch_size=args.batch_size, snr_list=snrs_to_process, mods_to_process=mods_to_process, limit=limit)

    # Group data based on specified SNRs and modulations
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Process the data and generate images for the specified SNRs and modulation types
    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            # Use tqdm to show progress for each modulation/SNR combination
            tqdm_desc = f'Processing {modulation_type} at SNR {snr}'
            for sample_idx, iq_data in tqdm(enumerate(samples), total=len(samples), desc=tqdm_desc):
                process_sample(iq_data, modulation_type, snr, sample_idx, 'constellation', (224, 224), image_types)
