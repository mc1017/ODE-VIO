import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def loadSequence(file_name):
    poses = []
    with open(file_name, "r") as f:
        for line in f:
            pose_flat = list(map(float, line.strip().split()))
            pose = np.zeros((4, 4))
            pose[:3, :4] = np.array(pose_flat).reshape((3, 4))
            pose[3, 3] = 1.0  # Assuming it's a homogeneous transformation matrix
            poses.append(pose)
    return np.array(poses)

def plot_combined_Path_2D(seq, percentage):
    """
    Plot ODE-VIO, VSVIO, DeepVO and GT path in 2D map
    """
    
    plot_path_dir = Path(f"/home/marco/Documents/NeuralCDE-VIO/results/GT-ODE-VIO-VSVIO-DeepVO/test/{percentage}%_dropout")
    vsvio_path_dir = Path(f"/home/marco/Documents/NeuralCDE-VIO/results/VSVIO-Original/test/{percentage}%_dropout")
    
    gt_path = plot_path_dir / f"{seq}_gt.txt"
    ode_path = plot_path_dir / f"{seq}_pred.txt"
    vsvio_path = vsvio_path_dir / f"{seq}_pred.txt"
    deepvo_path = plot_path_dir / f"{seq}_deepvo_pred.txt" 
    
    poses_gt_mat = loadSequence(gt_path) 
    poses_est_mat = loadSequence(ode_path)
    poses_vsvio_mat = loadSequence(vsvio_path)
    poses_deepvo_mat = loadSequence(deepvo_path)
    

    fontsize_ = 10
    title_fontsize = 12
    plot_keys = ["Ground Truth", "ODE-VIO (Ours)", "VSVIO", "DeepVO"]
    start_point = [0, 0]
    style_pred = "g-"
    style_gt = "b-"
    style_vsvio = "m-"
    style_deepvo = "b-"
    style_O = "ko"

    # get the value
    x_gt = np.asarray([pose[0, 3] for pose in poses_gt_mat])
    y_gt = np.asarray([pose[1, 3] for pose in poses_gt_mat])
    z_gt = np.asarray([pose[2, 3] for pose in poses_gt_mat])

    x_pred = np.asarray([pose[0, 3] for pose in poses_est_mat])
    y_pred = np.asarray([pose[1, 3] for pose in poses_est_mat])
    z_pred = np.asarray([pose[2, 3] for pose in poses_est_mat])
    
    z_vsvio_pred = np.asarray([pose[2, 3] for pose in poses_vsvio_mat])
    y_vsvio_pred = np.asarray([pose[1, 3] for pose in poses_vsvio_mat])
    x_vsvio_pred = np.asarray([pose[0, 3] for pose in poses_vsvio_mat])
    
    x_deepvo_pred = np.asarray([pose[0, 3] for pose in poses_deepvo_mat])
    y_deepvo_pred = np.asarray([pose[1, 3] for pose in poses_deepvo_mat])
    z_deepvo_pred = np.asarray([pose[2, 3] for pose in poses_deepvo_mat])

    # Plot 2d trajectory estimation map
    
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.gca()
    plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
    plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
    plt.plot(x_vsvio_pred, z_vsvio_pred, style_vsvio, label=plot_keys[2])
    plt.plot(-y_deepvo_pred, z_deepvo_pred, style_deepvo, label=plot_keys[3])
    plt.plot(start_point[0], start_point[1], style_O, label="Start Point")
    plt.legend(loc="upper right", prop={"size": fontsize_})
    plt.xlabel("x (m)", fontsize=title_fontsize)
    plt.ylabel("z (m)", fontsize=title_fontsize)
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

    plt.title("Sequnece 10", fontsize=title_fontsize)
    png_title = "{}_path_merged".format(seq)
    full_path = plot_path_dir / f"{seq}/{png_title}.png"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    

def main():
    dropout_ratio = 0.0
    plot_combined_Path_2D("10", int(dropout_ratio*100))
    


if __name__ == "__main__":
    main()
    
    
    