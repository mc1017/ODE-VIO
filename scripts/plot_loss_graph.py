# Convert loss file logs into a graph. 

import re
import matplotlib.pyplot as plt
import os
import numpy as np

# Path to your log file
log_file_path = '/home/marco/Documents/NeuralCDE-VIO/results/40/logs/train_40.txt'
save_path = "/home/marco/Documents/NeuralCDE-VIO/results/40/logs"  # Modify this to your actual path

# Lists to hold the extracted data
epochs = []
iterations = []
losses = []
eval_epochs = []
t_rel_values = []
r_rel_values = []

# Regular expression to match lines with loss values
loss_pattern =  re.compile(r'Epoch (\d+) training finished, pose loss: ([\d.]+)')
eval_pattern = re.compile(r"Epoch (\d+) evaluation finished, t_rel: (\d+\.\d+), r_rel: (\d+\.\d+),.*loss: (\d+\.\d+)")

# Read the log file and extract data
with open(log_file_path, 'r') as file:
    for line in file:
        match = loss_pattern.search(line)
        match2 = eval_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)
        if match2:
            eval_epochs.append(int(match.group(1)))
            t_rel_values.append(float(match.group(2)))
            r_rel_values.append(float(match.group(3)))

# Scatter plot for epochs vs loss, colored by epoch
plt.plot(epochs, losses, label='Pose Loss', marker='o', linestyle='-', markersize=2)  # Mark every 10th point

# Colorbar configuration
plt.xlabel('Epoch')
plt.ylabel('Pose Loss')
plt.title('Pose Loss Across Epochs')
plt.savefig(os.path.join(save_path, "pose_loss_by_epoch.png"))  # Save the figure
plt.grid(True)
plt.close()


# Plot the evaluation loss
plt.figure(figsize=(10, 6))
plt.plot(eval_epochs, t_rel_values, 'r-', label='translation_error')
plt.plot(eval_epochs, r_rel_values, 'b-', label='rotation_error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Evaluation Loss Curve')
plt.legend()
plt.savefig(os.path.join(save_path, "evaluation_loss_plot.png"))  # Save the figure
plt.close()  # Close the plot figure window