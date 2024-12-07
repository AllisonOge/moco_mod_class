import pickle
import h5py
import yaml
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

parser = argparse.ArgumentParser(description='Preprocess RML22 dataset')
parser.add_argument('--data', type=str, help='Path to RML22 dataset')


def main():
    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    snrs, mods = map(lambda j: sorted(
        list(set(map(lambda x: x[j], data.keys())))), [1, 0])

    X, y = [], []
    # convert modulation names to integers
    name2int = {mod: i for i, mod in enumerate(mods)}
    # save the mapping to a file
    with open('mod2int.yaml', 'w') as f:
        yaml.safe_dump(name2int, f)
    for mod in mods:
        for snr in snrs:
            X.append(data[(mod, snr)])
            y.extend([(name2int[mod], snr),]*len(data[(mod, snr)]))

    X = np.vstack(X)
    y = np.array(y)

    # print(X.shape, y.shape)

    # split dataset to train, validation and test (use stratifiedshufflesplit)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, test_idx = next(ss.split(X, y))
    devX, devy = X[dev_idx], y[dev_idx]
    testX, testy = X[test_idx], y[test_idx]
    train_idx, val_idx = next(ss.split(devX, devy))
    trainX, trainy = devX[train_idx], devy[train_idx]
    valX, valy = devX[val_idx], devy[val_idx]

    # describe the splits
    print(f"Train data has {trainX.shape[0]} samples")
    print(f"Validation data has {valX.shape[0]} samples")
    print(f"Test data has {testX.shape[0]} samples")
    print(f"The overall dataset has {X.shape[0]} samples")

    # save the dataset to h5 file
    with h5py.File("RML22_train.h5", "w") as f:
        f.create_dataset("data", data=trainX)
        f.create_dataset("label", data=trainy)

    with h5py.File("RML22_val.h5", "w") as f:
        f.create_dataset("data", data=valX)
        f.create_dataset("label", data=valy)

    with h5py.File("RML22_test.h5", "w") as f:
        f.create_dataset("data", data=testX)
        f.create_dataset("label", data=testy)


if __name__ == '__main__':
    main()
