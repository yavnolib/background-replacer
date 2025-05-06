import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [A.Resize(256, 256), A.HorizontalFlip(p=0.5), A.Normalize(), ToTensorV2()]
    )


def get_val_transforms():
    return A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
