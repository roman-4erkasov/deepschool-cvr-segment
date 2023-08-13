import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
import csv
from ast import literal_eval
import typing as tp
from src.utils import read_annotation, read_image, prepare_image


MAX_UINT8 = 255


class Dataset:
    def __init__(self, ann_path, images_path):
        self.ann_path = ann_path
        self.images_path = images_path
        self.ann = read_annotation(ann_path)

    def __getitem__(self, idx: int):
        image = read_image(
            os.path.join(
                self.images_path,
                self.ann[idx]["filename"]
            )
        )
        image = prepare_image(image)
        return {
            "image": image,
            "label": (
                *self.ann[idx]["p1"],
                *self.ann[idx]["p2"]
            ),
            "ocr": self.ann[idx]["code"],
        }
        
    def __len__(self) -> int:
        return len(self.ann)
