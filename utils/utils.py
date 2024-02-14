from pathlib import Path
import logging

def setup_experiment_directories(args):
    experiment_dir = Path('./results')
    experiment_dir.mkdir(exist_ok=True)

    file_dir = experiment_dir.joinpath(args.experiment_name)
    file_dir.mkdir(exist_ok=True)

    checkpoints_dir = file_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = file_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    return checkpoints_dir, log_dir


def setup_logger(args, log_dir):
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + f'/train_{args.experiment_name}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    return logger