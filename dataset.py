from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class ImageFolder(Dataset):
    def __init__(self, root, transform, crop=False):
        self.imgpaths = [os.path.join(root, x) for x in os.listdir(root)]
        self.transform = transform
        self.crop = crop

    def __getitem__(self, idx):
        path = self.imgpaths[idx]
        image = Image.open(path)
        width, height = image.size
        if self.crop:
            image = torchvision.transforms.functional.crop(image, 0, 0, int(4/5*height), width)
        return self.transform(image)

    def __len__(self):
        return len(self.imgpaths)
