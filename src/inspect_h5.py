# src/inspect_h5.py
import h5py

with h5py.File('data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5', 'r') as f:
    modulation_data = f['X'][:]
    print(f'Total frames for OOK at SNR -4.0: {modulation_data.shape[0]}')
