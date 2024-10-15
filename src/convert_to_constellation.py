# src/convert_to_constellation.py
import logging
from tqdm import tqdm
from data_loader import get_dataloader
from utils.constellation_data_processing_utils import process_sample, group_by_modulation_snr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    # Define SNRs and modulation types to process
    snrs_to_process = [20, 22, 24, 26, 28, 30]
    mods_to_process = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16QAM', '64QAM', '256QAM']
    limit = 32  # Set a limit if you want to restrict samples

    # Define image types, including "raw"
    image_types = ['grayscale']

    # Get the dataloader and process the data
    dataloader, mod2int = get_dataloader(batch_size=4096, snr_list=snrs_to_process, mods_to_process=mods_to_process, limit=limit)

    # Group data based on specified SNRs and modulations
    grouped_data, int2mod = group_by_modulation_snr(dataloader, mod2int)

    # Process the data and generate images for the specified SNRs and modulation types
    for modulation_type, snr_dict in grouped_data.items():
        for snr, samples in snr_dict.items():
            # Use tqdm to show progress for each modulation/SNR combination
            tqdm_desc = f'Processing {modulation_type} at SNR {snr}'
            for sample_idx, iq_data in tqdm(enumerate(samples), total=len(samples), desc=tqdm_desc):
                process_sample(iq_data, modulation_type, snr, sample_idx, 'constellation', (224, 224), image_types)
