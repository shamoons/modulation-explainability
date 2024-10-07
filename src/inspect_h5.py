import h5py


def inspect_h5_file(h5_file='data/RML2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5'):
    with h5py.File(h5_file, 'r') as f:
        print("Keys in the HDF5 file:")
        for key in f.keys():
            print(f"Key: {key}, Shape: {f[key].shape}")


if __name__ == '__main__':
    inspect_h5_file()
