from typing import Generator
import gcsfs
import pandas as pd
import numpy as np
import argparse
import h5py
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Read bucket and preprocess all files readable and save stats to csv file for analysis')
parser.add_argument('--recordings', type=str, nargs="+",
                    help='GCS bucket URLs to read recordings from (space-separated)', required=True)
parser.add_argument('--save_location', type=str,
                    help='Save location', default=".")
parser.add_argument('--name', type=str,
                    help='Name of the dataset', default="train")
parser.add_argument('--suffix', type=str,
                    help='Suffix of the dataset', default="sliced")
parser.add_argument('--length', type=int,
                    help='Length of the signal', default=1024)


def read_recording(file: str, fs: gcsfs.GCSFileSystem) -> tuple:
    """
    Handle error reading file with a try except block
    """
    with fs.open(file, 'rb') as f:
        data = np.load(f)
        _ = np.load(f, allow_pickle=True)
        metadata = np.load(f, allow_pickle=True).item()

    return data, metadata


def save_to_hdf5(filename: str, generator: Generator, length: int = 1024):
    """
    Save generator to hdf5 file
    """
    with h5py.File(filename, 'w') as f:
        data_dset = f.create_dataset(
            "data", (0, 2, length), maxshape=(None, 2, length), dtype='float32')

        for signal_batch in tqdm(generator, desc="Saving to HDF5", unit="batch"):
            data_dset.resize(data_dset.shape[0] + len(signal_batch), axis=0)
            data_dset[-len(signal_batch):] = signal_batch


def list_gcs_files(bucket_urls: list) -> list:
    """
    List all files in a bucket
    """
    fs = gcsfs.GCSFileSystem()
    all_files = []

    for bucket_url in bucket_urls:
        # Use walk to recursively traverse the directory structure
        for dirpath, _, filenames in fs.walk(bucket_url):
            # Add files that end with .npy
            all_files.extend(
                [f"{dirpath}/{file}" for file in filenames if file.endswith(".npy")])

    print(f"Found {len(all_files)} numpy files in {', '.join(bucket_urls)}")
    return all_files


def preprocess_file(file: str, fs: gcsfs.GCSFileSystem, length: int = 1024) -> Generator[np.ndarray, None, None]:
    """
    Slices the data into chunks based on desired length of signal
    """
    signal, _ = read_recording(file, fs)
    n_chunks = signal.shape[-1] // length
    slices = np.array(
        np.split(signal[:, :n_chunks * length], n_chunks, axis=-1))
    yield slices


def accumulate_files(files: list, fs: gcsfs.GCSFileSystem, length: int = 1024) -> Generator[np.ndarray, None, None]:
    """
    Accumulate all files in a generator
    """
    for file in tqdm(files, unit="file", desc="Preprocessing files"):
        try:
            yield from preprocess_file(file, fs, length)
        except ValueError as e:
            print(f"Error processing file {file}: {e}")


def main():
    args = parser.parse_args()

    fs = gcsfs.GCSFileSystem()
    dataset_path = f"{args.save_location}/{args.name}_{args.suffix}.h5"
    print("Storing preprocessed files to", dataset_path)

    recording_dir = list_gcs_files(args.recordings)

    if len(recording_dir) == 0:
        print("No files found in the bucket")
        return

    # save metadata of all files in bucket url
    metadata_all = []
    for file in tqdm(recording_dir, unit="file", desc="Reading metadata"):
        try:
            _, metadata = read_recording(file, fs)
            metadata_all.append(metadata)
        except ValueError as e:
            print(f"Error processing file {file}: {e}")

    metadata_df = pd.DataFrame(metadata_all)
    metadata_df.to_csv(
        f"{args.save_location}/{args.name}_{args.suffix}_metadata.csv")

    # save all files to hdf5
    save_to_hdf5(dataset_path, accumulate_files(
        recording_dir, fs, args.length), length=args.length)


if __name__ == "__main__":
    main()
