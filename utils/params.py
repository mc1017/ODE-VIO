import torch


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
