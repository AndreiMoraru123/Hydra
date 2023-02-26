import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os


class NYUDv2Dataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, "rb") as f:
            data = f.readlines()
        self.data = [line.decode("utf-8").strip("\n").split("\t") for line in data]
        self.transform = transform
        self.root_dir = "nyud"
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        abs_path = [os.path.join(self.root_dir, rpath) for rpath in self.data[idx]]
        sample = {"image": np.array(Image.open(abs_path[0]))}

        for msk_name, msk_path in zip(self.masks_names, abs_path[1:]):
            sample[msk_name] = np.array(Image.open(msk_path))

        sample["names"] = self.masks_names

        if self.transform:
            sample = self.transform(sample)

        return sample
