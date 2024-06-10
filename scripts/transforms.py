from src.data.utils import (
    ToTensor,
    Resize,
    RandomHorizontalFlip,
    RandomColorAug,
    Compose,
    NormalizeImage,
    NormalizeIMU,
)

def get_transforms(args):
    transform_train = [ToTensor(), Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [RandomHorizontalFlip()]
    if args.color:
        transform_train += [RandomColorAug()]
    if args.normalize:
        transform_train += [Compose(
            [
                NormalizeImage(mean=[0, 0, 0], std=[255, 255, 255]),
                NormalizeImage(mean=[0.45, 0.432, 0.411], std=[1, 1, 1]),
            ]
        )]
        transform_train += [NormalizeIMU(mean=[-0.06488193231511283, 0.07902796516539179, 9.79077591555693, 0.00014412904498676678, 0.0005592404262331839, -0.006576814886443332], 
        std=[1.0056579695115881, 1.2166065807036786, 0.403151671374919, 
            0.024120224040969432, 0.027277376120338145, 0.17162947412046847])]
        
    transform_train = Compose(transform_train)
    return transform_train