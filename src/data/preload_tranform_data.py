import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from src.data.utils import rotationError, read_pose_from_text, read_time_from_text
from pathlib import Path
import scipy.io as sio
import os
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--data_dir', type=str, default='/mnt/data0/marco/KITTI/data', help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
args = parser.parse_args()

# Define transformations
transform = Compose([
    Resize((args.img_h, args.img_w)), # Adjust H and W to your required dimensions
    ToTensor()
])

# Example paths
data_dir = Path(args.data_dir)
train_seqs = ['00', '01', '02', '04', '06', '08', '09']

SEQUENCE_LENGTH = 11
IMU_FEATURES = 6
GT_FEATURES = 16
TS_LENGTH = SEQUENCE_LENGTH
IMU_FREQ = 10
BATCH_SIZE = args.batch_size

# Count the numebr of datapoints
num_total_samples = 0
for folder in train_seqs:
    img_folder_path = data_dir / 'sequences' / folder / 'image_2'
    fpaths = list(img_folder_path.glob("*.png"))
    num_total_samples += len(fpaths)

# Initialize memory-mapped arrays
img_shape = (num_total_samples, 3, args.img_h, args.img_w)
imu_shape = (num_total_samples, IMU_FREQ * SEQUENCE_LENGTH, IMU_FEATURES)
pose_shape = (num_total_samples, SEQUENCE_LENGTH, 4, 4)  # Adjust based on your actual data structure
timestamp_shape = (num_total_samples, SEQUENCE_LENGTH)

# Initialize lists for data collection
for folder_idx, folder in enumerate(train_seqs):
    # Extraact pose data from text file
    poses, poses_rel = read_pose_from_text(data_dir/'poses/{}.txt'.format(folder))
    
    # Extract times information from text file
    timestamps = read_time_from_text(data_dir/'sequences/{}/times.txt'.format(folder))
    
    # Extracts imus data from matlab file with column 'imu_data_interp'
    imus = sio.loadmat(data_dir/'imus/{}.mat'.format(folder))['imu_data_interp']

    # Use glob method to find .png files
    fpaths = sorted((data_dir/'sequences/{}/image_2'.format(folder)).glob("*.png"))
    
    folder_data = []
    for i in range(len(fpaths)-SEQUENCE_LENGTH):
        img_samples = fpaths[i:i+SEQUENCE_LENGTH]
        for img_path in img_samples:
            image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
            image = transform(image)  # Apply transformations
            # Convert tensor to numpy array and store in memmap
            img_data[sample_idx] = image.numpy()
        # img_samples = no. sequence_len images
        
        timestamps_samples = timestamps[i:i+SEQUENCE_LENGTH]
        assert len(img_samples) == len(timestamps_samples)
        assert all(x < y for x, y in zip(timestamps, timestamps[1:])) # Check if timestamps are in ascending order
        # timestamps_samples.shape = (11, 1)
        
        imu_samples = imus[i*IMU_FREQ:(i+SEQUENCE_LENGTH-1)*IMU_FREQ+1]
        # imu_samples.shape = (101, 6), (i+SEQUENCE_LENGTH-1)*IMU_FREQ+1 = (0 + 11 - 1) * 10 + 1
        
        
        pose_samples = poses[i:i+SEQUENCE_LENGTH]
        # pose_samples.shape = (11, 4, 4), each 4x4 matrix
        
        pose_rel_samples = poses_rel[i:i+SEQUENCE_LENGTH-1]
        segment_rot = rotationError(pose_samples[0], pose_samples[-1])
        # Append processed samples to folder_data
        folder_data.append((img_samples, imu_samples, pose_samples, pose_rel_samples, timestamps_samples))
    

# Assuming img_samples is a list of tensors, get one sample to infer shape

# Make sure to flush the changes to disk
img_data.flush()
imu_data.flush()
gt_data.flush()
ts_data.flush()

print("Pre-processing complete.")
