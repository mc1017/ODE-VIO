from src.data.utils import (
    ToTensor,
    Resize,
    RandomHorizontalFlip,
    RandomColorAug,
    Compose,
)

def get_transforms(args):
    transform_train = [ToTensor(), Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [RandomHorizontalFlip()]
    if args.color:
        transform_train += [RandomColorAug()]
    transform_train = Compose(transform_train)
    return transform_train