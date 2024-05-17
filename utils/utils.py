from pathlib import Path
import logging
import torch
import sys


def setup_experiment_directories(args):
    experiment_dir = Path("./results")
    experiment_dir.mkdir(exist_ok=True)

    file_dir = experiment_dir.joinpath(args.experiment_name)
    file_dir.mkdir(exist_ok=True)

    checkpoints_dir = Path("/mnt/data0/marco/NeuralCDE-VIO")
    checkpoints_dir = checkpoints_dir.joinpath(args.experiment_name, "checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    log_dir = file_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True)

    graph_dir = file_dir.joinpath("graphs")
    graph_dir.mkdir(exist_ok=True)
    
    flownet_dir = file_dir.joinpath("flownets")
    flownet_dir.mkdir(exist_ok=True)
    
    img_dir = file_dir.joinpath("imgs")
    img_dir.mkdir(exist_ok=True) 
    
    return checkpoints_dir, log_dir, graph_dir, flownet_dir, img_dir


def setup_training_logger(args, log_dir):
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        str(log_dir) + f"/train_{args.experiment_name}.txt"
    )
     # Handler for printing to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(
        "----------------------------------------TRAINING----------------------------------"
    )
    logger.info("PARAMETER ...")
    logger.info(args)
    return logger

def setup_debug_logger(args, log_dir):
    # Create a new logger for dataset logging
    logger = logging.getLogger(f"{args.experiment_name}_debug")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all log messages

    # Ensure no duplicate handlers
    if not logger.handlers:
        # Create a formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Set up a file handler to write to a different log file
        file_handler = logging.FileHandler(str(log_dir) + f"/debug_{args.experiment_name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def print_tensor_stats(tensor, name, logger):
    """Print statistics for a given tensor."""
    """Log statistics for a given tensor."""
    logger.debug(f"Statistics for {name}:")
    logger.debug(f"  Device: {tensor.device}")
    logger.debug(f"  Data type: {tensor.dtype}")
    logger.debug(f"  Shape: {tensor.shape}")
    logger.debug(f"  Min value: {torch.min(tensor).item()}")
    logger.debug(f"  Max value: {torch.max(tensor).item()}")
    logger.debug(f"  Mean value: {torch.mean(tensor.float()).item()}")
    logger.debug(f"  Std deviation: {torch.std(tensor.float()).item()}")
    logger.debug(f"  Contains NaN: {torch.isnan(tensor).any().item()}")
    logger.debug(f"  Contains Inf: {torch.isinf(tensor).any().item()}")
