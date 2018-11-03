import os
import glob

import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

import util
from constant import *


class DatasetIterator(object):
    """Datasetを生成するクラス"""
    def __init__(self, csv_files, shuffle=True, **kwargs):
        self.csv_files = csv_files
        self.shuffle = shuffle
        self.kwargs = kwargs

    def __len__(self):
        return len(self.csv_files)

    def __iter__(self):
        if self.shuffle:
            csv_files = np.random.permutation(self.csv_files)
        else:
            csv_files = self.csv_files

        for path_csv in csv_files:
            dataset_train = QuickDrawDataset(path_csv, **self.kwargs)
            yield dataset_train


class DatasetManager(object):
    """DatasetIteratorを生成するクラス"""

    def __init__(self, dir_input):
        self.dir_input = dir_input

    def gen_train_and_valid(self, idx_kfold, kfold, **kwargs):

        def _get_subset_idx(filename):
            return int(os.path.splitext(os.path.basename(filename))[0].replace('train_k', ''))

        csv_files = sorted(glob.glob(os.path.join(self.dir_input, "*.csv")), key=_get_subset_idx)
        assert len(csv_files) > 0, "No csv file in {}".format(self.dir_input)

        csv_train, csv_valid = util.split_kfold(np.array(csv_files), idx_kfold, kfold)
        # print("Training files: {}".format(csv_train))
        # print("Validation files: {}".format(csv_valid))

        train_datasubset_generator = DatasetIterator(csv_train, shuffle=True, **kwargs)
        valid_datasubset_generator = DatasetIterator(csv_valid, shuffle=False, **kwargs)

        return train_datasubset_generator, valid_datasubset_generator


class _ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(image)
        return sample


class _Normalize(object):
    def __call__(self, sample):
        image = sample['image']
        # normalize to [0, 1]
        sample['image'] = image.float() / 255
        return sample


class QuickDrawDataset(Dataset):

    def __init__(self, path_csv, shape=(256, 256, 3), transform=transforms.Compose([_ToTensor(), _Normalize()]),
                 mode='train', thickness=6, draw_first=True):
        self.dataframe = pd.read_csv(path_csv)
        self.shape = shape
        self.transform = transform
        self.mode = mode
        self.thickness = thickness
        self.draw_first = draw_first

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        """
        train modeでは画像とカテゴリID、ｔｅｓｔ modeでは画像とkey_idを返す
        :param idx:
        :return:
        """
        row = self.dataframe.iloc[idx]
        strokes = util.parse_drawing(row['drawing'])
        image = draw_image(strokes, shape=self.shape, thickness=self.thickness, draw_first=self.draw_first)
        sample = {'image': image}
        if self.mode == 'train':
            sample['y'] = CATEGORY_ID_DICT[row['word']]
        elif self.mode == 'test':
            sample['key_id'] = row['key_id']

        if self.transform:
            sample = self.transform(sample)

        return sample


def draw_image(strokes, shape=(256, 256, 3), thickness=6, time_color=True, draw_first=True):
    if draw_first:
        return _draw_image_resize(strokes, shape, thickness, time_color)
    else:
        return _resize_draw_image(strokes, shape, thickness, time_color)


def _time_color(n_ch, t, time_color):
    if n_ch == 3:
        color = np.array((255, 255, 255)) - min(t, 10) * 13 if time_color else (255, 255, 255)
        color = tuple((int(c) for c in color))
    elif n_ch == 1:
        color = 255 - min(t, 10) * 13 if time_color else 255
    return color


def _resize_draw_image(strokes, shape=(256, 256, 3), thickness=1, time_color=True):
    """縮小してから描画"""
    image = np.zeros(shape=shape, dtype=np.uint8)
    w_max = np.max([np.max(s[0]) for s in strokes])
    h_max = np.max([np.max(s[1]) for s in strokes])
    w_offset = (ORIG_WIDTH - w_max) / 2
    h_offset = (ORIG_HEIGHT - h_max) / 2
    for t, stroke in enumerate(strokes):
        color = _time_color(shape[2], t, time_color)
        stroke[0] = (stroke[0] + w_offset) * shape[0] / ORIG_WIDTH
        stroke[1] = (stroke[1] + h_offset) * shape[1] / ORIG_HEIGHT
        stroke = np.transpose(np.asarray(stroke))
        if stroke.shape[1] == 3:
            stroke = stroke[:, :2]
        stroke = np.int32([stroke])
        cv2.polylines(image, stroke, isClosed=False, color=color, thickness=thickness)
    return image


def _draw_image_resize(strokes, shape=(256, 256, 3), thickness=6, time_color=True):
    """描画してから縮小"""
    img = np.zeros(shape=(ORIG_HEIGHT, ORIG_WIDTH, shape[2]), dtype=np.uint8)
    for t, stroke in enumerate(strokes):
        for i in range(len(stroke[0]) - 1):
            color = _time_color(shape[2], t, time_color)
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, thickness)
    if shape[0] != ORIG_HEIGHT or shape[1] != ORIG_WIDTH:
        return (127 - (255 - cv2.resize(img, shape[:2]))) / 127
    else:
        return (127 - (255 - img)) / 127


