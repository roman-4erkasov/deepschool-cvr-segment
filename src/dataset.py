import os
import cv2
from PIL import Image
import torch as th
import torchvision as thv
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import csv
from ast import literal_eval
import typing as tp
from src.utils import read_annotation, read_image, prepare_image


MAX_UINT8 = 255


class Dataset:
    def __init__(self, ann_path, images_path, task=None, transforms=None, dry_run=False):
        if task is None:
            task="detection"
        self.task = task
        self.ann_path = ann_path
        self.images_path = images_path
        self.ann = read_annotation(ann_path, dry_run)
        # self.to_pil = thv.transforms.ToPILImage()
        self.transforms = transforms
        
            

    def __getitem__(self, idx: int):
        # print(f"Dataset.__getitem__({idx=})")
        xmin, xmax = sorted(
            (self.ann[idx]["p1"][0], self.ann[idx]["p2"][0])
        )
        ymin, ymax = sorted(
            (self.ann[idx]["p1"][1], self.ann[idx]["p2"][1])
        )
        if self.task == "detection":
            # image = Image.open(
            #     os.path.join(
            #         self.images_path,
            #         self.ann[idx]["filename"]
            #     )
            # )
            image = read_image(
                os.path.join(
                    self.images_path, self.ann[idx]["filename"]
                )
            )  # HWC
            bboxes = [[xmin, ymin, xmax, ymax, "barcode"]]
            if self.transforms:
                transformed = self.transforms(image=image, bboxes=bboxes)
                image, target = transformed["image"], transformed["bboxes"]
            image = np.transpose(image,  (2, 0, 1))  # HWC->CHW
            image = image / MAX_UINT8
            image = th.FloatTensor(image)
            target = {
                "boxes": th.LongTensor([box[:-1] for box in bboxes]),
                # "ocr": self.ann[idx]["code"],
                "labels": th.LongTensor([1]),
            }
            result = image, target
        else:
            image = read_image(
                os.path.join(
                    self.images_path, self.ann[idx]["filename"]
                )
            )
            # image =  thv.transforms.functional.to_tensor(image)
            # image = Image.open(
            #     os.path.join(
            #         self.images_path,
            #         self.ann[idx]["filename"]
            #     )
            # )
            # image = np.transpose(image,  (2, 0, 1))  # HWC->CHW
            # image = image.astype(np.float32)
            
            # image = np.transpose(image,  (2, 0, 1))
            target = np.zeros(
                shape=[image.shape[0], image.shape[1], 1], 
                dtype=np.float32
            )
            target[xmin:xmax+1, ymin:ymax+1, 0] = 1.
            if self.transforms:
                transformed = self.transforms(image=image, mask=target)
                image, target = transformed["image"], transformed["mask"]
            image = np.transpose(image,  (2, 0, 1))
            target = np.transpose(target,  (2, 0, 1))
            result = {"image": image, "mask": target}
        return result
        
    def __len__(self) -> int:
        return len(self.ann)
