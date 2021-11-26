import torch
import h5py
from tqdm import tqdm


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, h5_filename):
        self.h5_filename = h5_filename
        self.h5f = h5py.File(self.h5_filename, "r")
        self.idx_to_key, self.num_examples = self.map_idx_to_key()

    def map_idx_to_key(self):

        num_examples = 0
        idx_to_key = {}

        for k in tqdm(sorted(self.h5f.keys(), key=lambda x: int(x[1:]))):
            assert k.startswith("X") or k.startswith("Y")

            if k.startswith("X"):
                for idx in range(self.h5f[k].shape[0]):
                    idx_to_key[idx + num_examples] = (k, idx)
                num_examples += self.h5f[k].shape[0]

        assert max(idx_to_key) == num_examples - 1
        return idx_to_key, num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        breakpoint()
        Xk, idx = self.idx_to_key[idx]
        Yk = Xk.replace("X", "Y")
        return self.h5f[Xk][idx], self.h5f[Yk][0, idx]


def test_H5Dataset():
    h5_filename = "data/dataset_train_all.h5"
    dataset = H5Dataset(h5_filename)

    for k in dataset.idx_to_key:
        print(dataset.idx_to_key[k])
        if k > 500:
            break
