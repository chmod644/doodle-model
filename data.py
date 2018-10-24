import os
import glob
from ast import literal_eval

import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from config import *


def draw_image(strokes, shape=(256, 256, 3), thickness=1):
    image = np.zeros(shape=shape, dtype=np.uint8)
    for stroke in strokes:
        stroke = np.transpose(np.array(stroke))
        if stroke.shape[1] == 3:
            stroke = stroke[:, :2]
        stroke = np.int32([stroke])
        cv2.polylines(image, stroke, isClosed=False, color=(255, 255, 255), thickness=thickness)
    return image


def filename_to_classname(filename):
    return os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")


class DatasetGenerator(object):

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
        assert len(self.csv_files) > 0, "No csv file in {}".format(self.input_dir)

    def _split_kfold(self, samples, idx_kfold, n_split):
        assert 0 <= idx_kfold < n_split
        assert len(samples) > n_split
        num_sample = len(samples)
        valid_index = range(idx_kfold, num_sample, n_split)
        train_index = list(set(range(num_sample)) - set(valid_index))
        samples_train = samples[train_index]
        samples_valid = samples[valid_index]
        return samples_train, samples_valid

    def gen_train_and_valid(self, shape, idx_kfold, n_split):
        df_train = []
        df_valid = []
        for f in self.csv_files:
            class_name = filename_to_classname(f)
            id_class = CLASS_NAMES.index(class_name)
            df = pd.read_csv(f, index_col='key_id')
            df['id_class'] = id_class
            df = df.sort_values("countrycode")
            index_train, index_valid = self._split_kfold(np.array(df.index.tolist()), idx_kfold, n_split)
            df_train_local = df.loc[index_train]
            df_valid_local = df.loc[index_valid]
            df_train.append(df_train_local)
            df_valid.append(df_valid_local)
            break
        df_train = pd.concat(df_train)
        df_valid = pd.concat(df_valid)

        dataset_train = DoodleDataset(df_train, transform=transforms.Compose([_ToTensor()]))
        dataset_valid = DoodleDataset(df_valid, transform=transforms.Compose([_ToTensor()]))

        return dataset_train, dataset_valid


class DoodleDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]
        strokes = literal_eval(row['drawing'])
        image = draw_image(strokes)
        id_class = row['id_class']

        sample = {'id_class': id_class, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class _ToTensor(object):
    def __call__(self, sample):
        id_class, image = sample['id_class'], sample['image']
        image = image.transpose((2, 0, 1))
        return {'id_class': id_class, 'image': torch.from_numpy(image)}
