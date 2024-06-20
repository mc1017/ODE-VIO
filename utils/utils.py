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

def set_gpu_ids(args):
    str_ids = args.gpu_ids.split(",")
    gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available.")
        # Get the current default CUDA device
        current_device = torch.cuda.current_device()
        print("Current CUDA Device:", torch.cuda.get_device_name(current_device))
    else:
        print("CUDA is not available. Using CPU.")
    return gpu_ids[0]


def load_pretrained_model(model, args):
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print(f"Loaded model from {args.pretrain}")
    else:
        print("Training from scratch")


def get_optimizer(model, args):
    param_groups = [
        {'params': model.module.Pose_net.get_other_params(), 'lr': args.lr_warmup},
        {'params': model.module.Pose_net.get_regressor_params(), 'lr': args.lr_warmup}
    ]
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(param_groups, lr=1e-4, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=args.lr_warmup,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
        )
    return optimizer