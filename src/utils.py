import os
import cv2
import numpy as np
import torchvision as thv
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import csv
from ast import literal_eval
import typing as tp

MAX_UINT8 = 255


class ToTensor(object):
    """
    torchvision/reference/detection
    """
    def __call__(self, image, target):
        image =  thv.transforms.functional.to_tensor(image)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = thv.transforms.functional.resize(
            image, size=self.size
        )
        mask = thv.transforms.functional.resize(
            mask, size=self.size
        )
        return image, mask
        

class Compose(object):
    """
    torchvision/reference/detection
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transforms(train=None):
    """
    torchvision/reference/detection
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    # if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
    return thv.transforms.Compose(transforms)

def split_3way(
    ann_path, 
    target_dir,
    train_prop=0.8, 
    val_prop=0.1,
):
    d_full = pd.read_csv(ann_path, delimiter="\t")
    splits = (
        int(train_prop * d_full.shape[0]),
        int((train_prop+val_prop) * d_full.shape[0])
    )
    i_train, i_val, i_test = np.split(
        list(d_full.index), splits
    )
    for name, ids in [
        ("train", i_train),
        ("val", i_val),
        ("test", i_test),
    ]:
        df = d_full.loc[ids]
        df.to_csv(
            os.path.join(target_dir, f"{name}.tsv"),
            sep="\t",
            index=None,
            header=True
        )


def read_annotation(path: str, dry_run: bool = False, dry_run_limit: int = 5):
    result = []
    with open(path, newline='') as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for idx, row in enumerate(reader):
            if dry_run and dry_run_limit < idx:
                break
            row["p1"] = literal_eval(row["p1"])
            row["p2"] = literal_eval(row["p2"])
            result.append(row)
    return result


def read_image(image_path: str):
    """
    :param image_path: path to the image
    """
    # 1. reading file to np.ndarray with shape=(H,W,C)
    image = cv2.imread(image_path)
    # 2.  BGR -> RGB (OpenCV->common)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    

def resize_coord(coord, size_src, size_dst):
    x, y = coord
    x_src, y_src = size_src
    x_dst, y_dst = size_dst
    return x*x_dst/x_src, y*y_dst/y_src


def resize_coords(coords, size_src, size_dst):
    return [
        resize_coord(coord, size_src, size_dst)
        for coord in coords
    ]



def prepare_image(
    image,
    means=(0.485, 0.456, 0.406),
    stds=(0.229, 0.224, 0.225),
    target_image_size = (256, 256),
    bboxes=None
):
    """
    :param image: 
    :param means: mean values, 
        default - mean colors in ImageNet
    :param stds: standard deviations, 
        default - standard deviations of colors in ImageNet
    :param target_image_size:
    """
    # 6. scaling into interval [0,1]
    img = cv2.resize(image, target_image_size) / MAX_UINT8
    # 7. [H,W,C] -> [C,H,W] (common->torch-ish)
    img = np.transpose(img,  (2, 0, 1))
    # 8. substract mean values
    img -= np.array(means)[:, None, None]
    # 9. divide by standard deviations
    img /= np.array(stds)[:, None, None]
    return img

def draw_record(images_path, record):
    p1 = record["p1"]
    p2 = record["p2"]
    y = [p1[0], p2[0]]
    x = [p1[1], p2[1]]
    coords = [
        (min(x), min(y)),
        (min(x), max(y)),
        (max(x), max(y)),
        (max(x), min(y)),
        (min(x), min(y))
    ]
    print(f"{coords=}")
    poly = Polygon(coords, fc='none', ec='orangered')
    plt.imshow(image)
    plt.gca().add_patch(poly)


def draw_pth(
    image_pth, 
    p1, 
    p2,
    means=(0.485, 0.456, 0.406),
    stds=(0.229, 0.224, 0.225)
):
    img = image_pth * np.array(stds)[:, None, None]
    img += np.array(means)[:, None, None]
    img = np.transpose(img,  (1, 2, 0))
    img *= MAX_UINT8
    img
    
    