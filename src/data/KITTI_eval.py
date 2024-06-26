import os
import glob
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import random


from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from src.data.utils import (
    read_pose_from_text,
    read_time_from_text,
    path_accu,
    rmse_err_cal,
    trajectoryDistances,
    rotationError,
    translationError,
    computeOverallErr,
    saveSequence,
    lastFrameFromSegmentLength,
    concatenate_pose_changes,
)

IMU_FREQ = 10

class data_partition:
    def __init__(self, opt, folder):
        super(data_partition, self).__init__()
        self.opt = opt
        self.data_dir = opt.data_dir
        self.seq_len = opt.seq_len
        self.folder = folder
        self.dropout = opt.eval_data_dropout
        self.load_data()

    def load_data(self):
        image_dir = self.data_dir + "/sequences/"
        imu_dir = self.data_dir + "/imus/"
        pose_dir = self.data_dir + "/poses/"
        timestamp_dir = self.data_dir + "/sequences/"

        self.img_paths = glob.glob("{}{}/image_2/*.png".format(image_dir, self.folder))
        self.imus = sio.loadmat("{}{}.mat".format(imu_dir, self.folder))[
            "imu_data_interp"
        ]
        self.poses, self.poses_rel = read_pose_from_text(
            "{}{}.txt".format(pose_dir, self.folder)
        )
        self.timestamps = read_time_from_text(
            "{}{}/times.txt".format(timestamp_dir, self.folder)
        )
        self.img_paths.sort()
        
        
        # Create Irregularity in the data by dropping some data points
        i = 1 
        while i < len(self.poses_rel) - 2:
            if random.random() < self.dropout:
                self.poses_rel[i] = concatenate_pose_changes(self.poses_rel[i], self.poses_rel[i + 1])
                self.poses_rel = np.delete(self.poses_rel, i + 1, axis=0)
                self.poses = np.delete(self.poses, i, axis=0)
                self.timestamps = np.delete(self.timestamps, i, axis=0)
                self.imus = np.delete(self.imus, np.concatenate([np.arange(i * IMU_FREQ, (i + 1) * IMU_FREQ)]), axis=0)
                self.img_paths.pop(i)
            else:
                i += 1

        self.img_paths_list, self.poses_list, self.imus_list, self.timestamps_list = (
            [],
            [],
            [],
            [],
        )
        start = 0
        n_frames = len(self.img_paths)
        while start + self.seq_len < n_frames:
            self.img_paths_list.append(self.img_paths[start : start + self.seq_len])
            self.poses_list.append(self.poses_rel[start : start + self.seq_len - 1])
            self.timestamps_list.append(self.timestamps[start : start + self.seq_len])
            self.imus_list.append(
                self.imus[start * 10 : (start + self.seq_len - 1) * 10 + 1]
            )
            start += self.seq_len - 1
        self.img_paths_list.append(self.img_paths[start:])
        self.poses_list.append(self.poses_rel[start:])
        self.timestamps_list.append(self.timestamps[start:])
        self.imus_list.append(self.imus[start * 10 :])
        

    def __len__(self):
        return len(self.img_paths_list)

    def __getitem__(self, i):
        image_path_sequence = self.img_paths_list[i]
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_img = TF.resize(img_as_img, size=(self.opt.img_h, self.opt.img_w))
            img_as_tensor = TF.to_tensor(img_as_img.copy()) - 0.5
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        imu_sequence = torch.FloatTensor(self.imus_list[i])
        timestamp_sequence = torch.FloatTensor(self.timestamps_list[i])
        gt_sequence = self.poses_list[i][:, :6]
        return image_sequence, imu_sequence, gt_sequence, timestamp_sequence


class KITTI_tester:
    def __init__(self, args):
        super(KITTI_tester, self).__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))

        self.args = args

    def test_one_path(self, net, df, num_gpu=1):
        hc = None
        pose_list = []
        # with profile(activities=[
        # ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof:
        total_time = 0
        for i, (image_seq, imu_seq, gt_seq, ts_seq) in tqdm(
            enumerate(df), total=len(df), smoothing=0.9
        ):
            x_in = image_seq.unsqueeze(0).cuda().float()
            i_in = imu_seq.unsqueeze(0).cuda().float()
            t_in = ts_seq.unsqueeze(0).cuda().float()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                # with record_function("model_inference"):
                # start.record()
                pose, hc = net(x_in, i_in, t_in, hc=hc)
                # end.record()
            # zero_pose = torch.zeros_like(pose[:, :1, :])
            # padded_poses = torch.cat((zero_pose, pose[:, :-1, :]), dim=1)
            # relative_pose = pose - padded_poses
            # torch.cuda.synchronize()
            # elapsed_time_ms = start.elapsed_time(end)
            # print(f'Elapsed time: {elapsed_time_ms:.3f} ms')
            # total_time += elapsed_time_ms
            relative_pose = pose
            pose_list.append(relative_pose[0, :, :].detach().cpu().numpy())
            # decision_list.append(decision[0,:,:].detach().cpu().numpy()[:, 0])
            # probs_list.append(probs[0,:,:].detach().cpu().numpy())

        pose_est = np.vstack(pose_list)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
        print(total_time)
        return pose_est, None, None

    def eval(self, net, num_gpu=1):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f"testing sequence {seq}")
            pose_est, dec_est, prob_est = self.test_one_path(
                net, self.dataloader[i], num_gpu=num_gpu 
            )
            (
                pose_est_global,
                pose_gt_global,
                t_rel,
                r_rel,
                t_rmse,
                r_rmse,
                usage,
                speed,
            ) = kitti_eval(pose_est, dec_est, self.dataloader[i].poses_rel)

            self.est.append(
                {
                    "pose_est_global": pose_est_global,
                    "pose_gt_global": pose_gt_global,
                    "decs": dec_est,
                    "probs": prob_est,
                    "speed": speed,
                }
            )
            self.errors.append(
                {
                    "t_rel": t_rel,
                    "r_rel": r_rel,
                    "t_rmse": t_rmse,
                    "r_rmse": r_rmse,
                    "usage": usage,
                }
            )

        return self.errors

    def generate_plots(self, save_dir, ep):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(
                seq,
                ep,
                self.est[i]["pose_gt_global"],
                self.est[i]["pose_est_global"],
                save_dir,
                self.est[i]["decs"],
                self.est[i]["speed"],
            )

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir / "{}_pred.txt".format(seq)
            gt_path = save_dir / "{}_gt.txt".format(seq)
            saveSequence(self.est[i]["pose_est_global"], path)
            saveSequence(self.est[i]["pose_gt_global"], gt_path)
            print("Seq {} saved".format(seq))


def kitti_eval(pose_est, dec_est, pose_gt):

    # First decision is always true
    # dec_est = np.insert(dec_est, 0, 1)

    # Calculate the translational and rotational RMSE
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)
    # print("differences:", pose_est-pose_gt)
    # Transfer to 3x4 pose matrix
    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)
    # print("pose_est_mat:", pose_est_mat)
    # print("pose_gt_mat:", pose_gt_mat)

    # Using KITTI metric
    err_list, t_rel, r_rel, speed = kitti_err_cal(pose_est_mat, pose_gt_mat)

    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180
    # usage = np.mean(dec_est) * 100
    usage = 0

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, usage, speed


def kitti_err_cal(pose_est_mat, pose_gt_mat):

    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):

        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            # Continue if sequence not long enough
            if (
                last_frame == -1
                or last_frame >= len(pose_est_mat)
                or first_frame >= len(pose_est_mat)
            ):
                continue

            pose_delta_gt = np.dot(
                np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame]
            )
            pose_delta_result = np.dot(
                np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame]
            )

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)


def plotPath_2D(seq, ep, poses_gt_mat, poses_est_mat, plot_path_dir, decision, speed):

    # Apply smoothing to the decision
    # decision = np.insert(decision, 0, 1)
    # decision = moving_average(decision, window_size)

    fontsize_ = 10
    plot_keys = ["Ground Truth", "Ours"]
    start_point = [0, 0]
    style_pred = "b-"
    style_gt = "r-"
    style_O = "ko"

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])

    # Plot 2d trajectory estimation map
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(start_point[0], start_point[1], style_O, label="Start Point")
    plt.legend(loc="upper right", prop={"size": fontsize_})
    plt.xlabel("x (m)", fontsize=fontsize_)
    plt.ylabel("z (m)", fontsize=fontsize_)
    # set the range of x and y
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    plot_radius = max(
        [
            abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean), (ylim, ymean))
            for lim in lims
        ]
    )
    ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    plt.title("2D path")
    png_title = "{}_path_2d".format(seq)
    full_path = plot_path_dir / f"{seq}/{png_title}_{ep}.png"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Plot decision hearmap
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # ax = plt.gca()
    # # cout = np.insert(decision, 0, 0) * 100
    # # cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    # plt.xlabel('x (m)', fontsize=fontsize_)
    # plt.ylabel('z (m)', fontsize=fontsize_)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # xmean = np.mean(xlim)
    # ymean = np.mean(ylim)
    # ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    # ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    # max_usage = max(cout)
    # min_usage = min(cout)
    # ticks = np.floor(np.linspace(min_usage, max_usage, num=5))
    # cbar = fig.colorbar(cax, ticks=ticks)
    # cbar.ax.set_yticklabels([str(i) + '%' for i in ticks])

    # plt.title('decision heatmap with window size {}'.format(window_size))
    # png_title = "{}_decision_smoothed".format(seq)
    # full_path = plot_path_dir / f"{png_title}.png"
    # plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1)
    # plt.close()

    # # Plot the speed map
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # ax = plt.gca()
    # cout = speed
    # cax = plt.scatter(x_pred, z_pred, marker='o', c=cout)
    # plt.xlabel('x (m)', fontsize=fontsize_)
    # plt.ylabel('z (m)', fontsize=fontsize_)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # xmean = np.mean(xlim)
    # ymean = np.mean(ylim)
    # ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
    # ax.set_ylim([ymean - plot_radius, ymean + plot_radius])
    # max_speed = max(cout)
    # min_speed = min(cout)
    # ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    # cbar = fig.colorbar(cax, ticks=ticks)
    # cbar.ax.set_yticklabels([str(i) + 'm/s' for i in ticks])

    # plt.title('speed heatmap')
    # png_title = "{}_speed".format(seq)
    # full_path = plot_path_dir / f"{png_title}.png"
    # plt.savefig(full_path, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
