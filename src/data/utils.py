import os
import glob
import numpy as np
import time
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
import random
plt.switch_backend('agg')

_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R):
    '''
    check whether a matrix is a qualified rotation metrix
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices 
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def rotationError(Rt1, Rt2):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(Rt1, Rt2):
    '''
    Calculate the translational difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def eulerAnglesToRotationMatrix(theta):
    '''
    Calculate the rotation matrix from eular angles (roll, yaw, pitch)
    '''
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def normalize_angle_delta(angle):
    '''
    Normalization angles to constrain that it is between -pi and pi
    '''
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def pose_6DoF_to_matrix(pose):
    '''
    Calculate the 3x4 transformation matrix from Eular angles and translation vector
    '''
    R = eulerAnglesToRotationMatrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    R = np.concatenate((R, t), 1)
    R = np.concatenate((R, np.array([[0, 0, 0, 1]])), 0)
    return R

def pose_accu(Rt_pre, R_rel):
    '''
    Calculate the accumulated pose from the latest pose and the relative rotation and translation
    '''
    Rt_rel = pose_6DoF_to_matrix(R_rel)
    return Rt_pre @ Rt_rel

def path_accu(pose):
    '''
    Generate the global pose matrices from a series of relative poses
    '''
    answer = [np.eye(4)]
    for index in range(pose.shape[0]):
        pose_ = pose_accu(answer[-1], pose[index, :])
        answer.append(pose_)
    return answer

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def rmse_err_cal(pose_est, pose_gt):
    '''
    Calculate the rmse of relative translation and rotation
    '''
    t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
    return t_rmse, r_rmse

def trajectoryDistances(poses):
    '''
    Calculate the distance and speed for each frame
    '''
    dist = [0]
    speed = [0]
    for i in range(len(poses) - 1):
        cur_frame_idx = i
        next_frame_idx = cur_frame_idx + 1
        P1 = poses[cur_frame_idx]
        P2 = poses[next_frame_idx]
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dz = P1[2, 3] - P2[2, 3]
        dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        speed.append(np.sqrt(dx**2 + dy**2 + dz**2) * 10)
    return dist, speed

def lastFrameFromSegmentLength(dist, first_frame, len_):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + len_):
            return i
    return -1

def computeOverallErr(seq_err):
    t_err = 0
    r_err = 0
    seq_len = len(seq_err)

    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err

# With the 3x4 transformation matrix, we add 1 row to the bottom, making it a 
# 4x4 homogeneous Matrix
def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt
   
# Poses information is stored as a N x 12 table, where N is the number of
# frames of this sequence. Row i represents the i'th pose of the left camera
# coordinate system (i.e., z pointing forwards) via a 3x4 transformation
# matrix.

def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)            
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
        
    return poses_abs, poses_rel

def read_time_from_text(path):
    timestamps = []
    with open(path) as f:
        for line in f.readlines():
            t = float(line)
            timestamps.append(t)
    # print(timestamps)
    assert all(x < y for x, y in zip(timestamps, timestamps[1:]))
    return timestamps

def saveSequence(poses, file_name):
    with open(file_name, 'w') as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            f.write(' '.join([str(r) for r in pose]))
            f.write('\n')


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, imus, intrinsics, timestamps):
        for t in self.transforms:
            images, imus, intrinsics, timestamps = t(images, imus, intrinsics, timestamps)
        return images, imus, intrinsics, timestamps
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, imus, intrinsics, timestamps):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, imus, intrinsics, timestamps
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean: {}, '.format(self.mean)
        format_string += 'std: {})\n'.format(self.std)
        return format_string

class ToTensor(object):
    def __call__(self, images, imus, gts, timestamps):
        tensors = []
        for im in images:
            tensors.append(TF.to_tensor(im.copy())- 0.5)
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts, timestamps
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(object):
    def __init__(self, size=(256, 512)):
        self.size = size

    def __call__(self, images, imus, gts, timestamps):
        tensors = [TF.resize(im, size=self.size) for im in images]
        tensors = torch.stack(tensors, 0)
        return tensors, imus, gts, timestamps
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'img_h: {}, '.format(self.size[0])
        format_string += 'img_w: {})'.format(self.size[1])
        return format_string

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, images, imus, gts, timestamps):
        if random.random() < self.p:
            tensors = [TF.hflip(im) for im in images]
            tensors = torch.stack(tensors, 0)
            # Adjust imus and target poses according to horizontal flips
            imus[:, 1], imus[:, 3], imus[:, 5] = -imus[:, 1], -imus[:, 3], -imus[:, 5]
            gts[:, 1], gts[:, 2], gts[:, 3] = -gts[:, 1], -gts[:, 2], -gts[:, 3]
        else:
            tensors = images
        return tensors, imus, gts, timestamps
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p: {})'.format(self.p)
        return format_string

class RandomColorAug(object):
    def __init__(self, augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2], p=0.5):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2
        self.p = p
    def __call__(self, images, imus, gts, timestamps):
        if random.random() < self.p:
            images = images + 0.5
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            
            # randomly shift gamma
            img_aug = images ** random_gamma
            
            # randomly shift brightness
            img_aug = img_aug * random_brightness
            
            # randomly shift color
            for i in range(3):
                img_aug[:, i, :, :] *= random_colors[i]
            
            # saturate
            img_aug = torch.clamp(img_aug, 0, 1) - 0.5

        else:
            img_aug = images

        return img_aug, imus, gts, timestamps
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'gamma: {}-{}, '.format(self.gamma_low, self.gamma_high)
        format_string += 'brightness: {}-{}, '.format(self.brightness_low, self.brightness_high)
        format_string += 'color shift: {}-{}, '.format(self.color_low, self.color_high)
        format_string += 'p: {})'.format(self.p)
        return format_string
