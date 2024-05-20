import sys

sys.path.append("..")
import numpy as np
import random
import logging
from PIL import Image
from torch.utils.data import Dataset, BatchSampler
import scipy.io as sio
from pathlib import Path
from src.data.utils import rotationError, read_pose_from_text, read_time_from_text, concatenate_pose_changes
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d


IMU_FREQ = 10

class KITTI(Dataset):
    def __init__(
        self,
        root,
        sequence_length=11,
        train_seqs=["00", "01", "02", "04", "06", "08", "09"],
        transform=None,
        logger=None,
        dropout=0.0,
    ):

        self.root = Path(root)
        self.sequence_length = sequence_length  # This is sequence length for LSTM
        self.transform = transform
        self.train_seqs = train_seqs
        self.logger = logger
        self.dropout = dropout
        self.img_seq_len = [] # Length of each seqeunce after dropout
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []

        for folder in self.train_seqs:
            # Extraact pose data from text file
            poses, poses_rel = read_pose_from_text(
                self.root / "poses/{}.txt".format(folder)
            )

            # Extract times information from text file
            timestamps = read_time_from_text(
                self.root / "sequences/{}/times.txt".format(folder)
            )

            # Extracts imus data from matlab file with column 'imu_data_interp'
            imus = sio.loadmat(self.root / "imus/{}.mat".format(folder))[
                "imu_data_interp"
            ]
            # Use glob method to find .png files
            fpaths = sorted(
                (self.root / "sequences/{}/image_2".format(folder)).glob("*.png")
            )
            
            # Create Irregularity in the data by dropping some data points
            i = 1 
            while i < len(poses_rel) - 2:
                if random.random() < self.dropout:
                    poses_rel[i] = concatenate_pose_changes(poses_rel[i], poses_rel[i + 1])
                    poses_rel = np.delete(poses_rel, i + 1, axis=0)
                    poses = np.delete(poses, i, axis=0)
                    timestamps = np.delete(timestamps, i, axis=0)
                    imus = np.delete(imus, np.concatenate([np.arange(i * IMU_FREQ, (i + 1) * IMU_FREQ)]), axis=0)
                    fpaths.pop(i)
                else:
                    i += 1
            
            self.img_seq_len.append(len(fpaths))
            for i in range(0, len(fpaths) - self.sequence_length):
                img_samples = fpaths[i : i + self.sequence_length]
                # img_samples = no. sequence_len images

                timestamps_samples = timestamps[i : i + self.sequence_length]
                assert len(img_samples) == len(timestamps_samples)
                assert all(
                    x < y for x, y in zip(timestamps, timestamps[1:])
                )  # Check if timestamps are in ascending order
                # timestamps_samples.shape = (11, 1)

                imu_samples = imus[
                    i * IMU_FREQ : (i + self.sequence_length - 1) * IMU_FREQ + 1
                ]
                # imu_samples.shape = (101, 6), (i+self.sequence_length-1)*IMU_FREQ+1 = (0 + 11 - 1) * 10 + 1

                pose_samples = poses[i : i + self.sequence_length]
                # pose_samples.shape = (11, 4, 4), each 4x4 matrix

                pose_rel_samples = poses_rel[i : i + self.sequence_length - 1]
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {
                    "imgs": img_samples,
                    "imus": imu_samples,
                    "gts": pose_rel_samples,
                    "rot": segment_rot,
                    "timestamps": timestamps_samples,
                    "folder": folder,
                }
                sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample["imgs"]]
        imus = np.copy(sample["imus"])
        gts = np.copy(sample["gts"]).astype(np.float32)
        timestamps = np.copy(sample["timestamps"]).astype(np.float32)
        folder = sample["folder"]
        
        if self.transform is not None:
            imgs, imus, gts, timestamps = self.transform(imgs, imus, gts, timestamps)
        assert np.all(
            np.diff(timestamps) > 0
        ), "Timestamps must be strictly ascending - getitem"
        return imgs, imus, gts, timestamps, folder

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Training sequences: "
        for seq in self.train_seqs:
            fmt_str += "{} ".format(seq)
        fmt_str += "\n"
        fmt_str += "    Number of segments: {}\n".format(self.__len__())
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        )

    return kernel_window


# To create sequence batch sampler that does not batch across samples from other sequnces.
class SequenceBoundarySampler(BatchSampler):
    def __init__(
        self,
        root,
        batch_size,
        train_seqs=["00", "01", "02", "04", "06", "08", "09"],
        seq_len=11,
        shuffle=True,
        img_seq_length=None
    ):
        self.root = Path(root)
        self.seq_len = seq_len  # lstm seq length
        self.img_seq_len = self._find_img_seq_len(train_seqs) if not img_seq_length else img_seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = self._create_samples()
        self.batches = self._create_batches() 

    def _find_img_seq_len(self, train_seqs):
        img_seq_len = []
        for seq in train_seqs:
            fpaths = sorted(
                (self.root / "sequences/{}/image_2".format(seq)).glob("*.png")
            )
            img_seq_len.append(len(fpaths))
        return img_seq_len

    def _create_samples(self):
        samples = []
        for seq_idx, img_length in enumerate(self.img_seq_len):
            num_samples = img_length - self.seq_len
            for i in range(num_samples):
                samples.append((seq_idx, i))
        return samples

    def _create_batches(self):
        if self.shuffle:
            random.shuffle(self.samples)  # Shuffle all samples
            print("Shuffling Batches...")

        batches = []
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i : i + self.batch_size]
            batches.append(batch)
        return batches

    def __iter__(self):
        self.batches = self._create_batches()

        for batch in self.batches:
            yield [idx for _, idx in batch]

    def __len__(self):
        return len(self.batches)
