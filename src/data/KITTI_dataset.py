import sys

sys.path.append("..")
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, BatchSampler
import scipy.io as sio
from pathlib import Path
from src.data.utils import rotationError, read_pose_from_text, read_time_from_text
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
    ):

        self.root = Path(root)
        self.sequence_length = sequence_length  # This is sequence length for LSTM
        self.transform = transform
        self.train_seqs = train_seqs
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
        print("Samples number:", len(self.samples))

        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to the method in https://github.com/YyzHarry/imbalanced-regression
        rot_list = np.array(
            [np.cbrt(item["rot"] * 180 / np.pi) for item in self.samples]
        )
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [
            num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range) + 1)
        ]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel="gaussian", ks=7, sigma=5)
        eff_label_dist = convolve1d(
            np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
        )

        self.weights = [
            np.float32(1 / eff_label_dist[bin_idx - 1]) for bin_idx in indexes
        ]

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
        # print("Passed Getitem")

        rot = sample["rot"].astype(np.float32)
        weight = self.weights[index]

        return imgs, imus, gts, rot, weight, timestamps, folder

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
    ):
        self.root = Path(root)
        self.seq_len = seq_len  # lstm seq length
        self.img_seq_len = self._find_img_seq_len(train_seqs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = self._create_samples()
        self.batches = self._create_batches()
        
        print(self.batches)

    def _find_img_seq_len(self, train_seqs):
        img_seq_len = []
        for seq in train_seqs:
            fpaths = sorted(
                (self.root / "sequences/{}/image_2".format(seq)).glob("*.png")
            )
            img_seq_len.append(len(fpaths))
        return img_seq_len

    # def _create_batches(self):
    #     batches = []
    #     batch = []
    #     curr_idx = 0
    #     for seq_idx, img_length in enumerate(self.img_seq_len):

    #         # To find out how many samples there actually are in an image sequence, this formula calculates it directly
    #         num_samples = img_length - self.seq_len
    #         for _ in range(num_samples):
    #             batch.append(
    #                 (seq_idx, curr_idx)
    #             )  # (sequence index, element index of samples)
    #             if len(batch) == self.batch_size:
    #                 batches.append(batch)
    #                 batch = []
    #             curr_idx += 1
    #         if batch:
    #             batches.append(batch)
    #             batch = []

    #     return batches
    
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
        
        batches = []
        for i in range(0, len(self.samples), self.batch_size):
            batch = self.samples[i:i + self.batch_size]
            batches.append(batch)
        
        return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
            
        for batch in self.batches:
            yield [idx for _, idx in batch]

    def __len__(self):
        return len(self.batches)
