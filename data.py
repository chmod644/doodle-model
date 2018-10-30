import os
import glob
from ast import literal_eval

import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from config import *
from util import filename_to_classname


def draw_image(strokes, shape=(256, 256, 3), thickness=1):
    image = np.zeros(shape=shape, dtype=np.uint8)
    w_max = np.max([np.max(s[0]) for s in strokes])
    h_max = np.max([np.max(s[1]) for s in strokes])
    w_offset = (ORIG_WIDTH - w_max) / 2
    h_offset = (ORIG_HEIGHT - h_max) / 2
    for stroke in strokes:
        stroke[0] = (stroke[0] + w_offset) * shape[0] / ORIG_WIDTH
        stroke[1] = (stroke[1] + h_offset) * shape[1] / ORIG_HEIGHT
        stroke = np.transpose(np.asarray(stroke))
        if stroke.shape[1] == 3:
            stroke = stroke[:, :2]
        stroke = np.int32([stroke])
        cv2.polylines(image, stroke, isClosed=False, color=(255, 255, 255), thickness=thickness)
    return image


class DatasetGenerator(object):

    def __init__(self, dir_input):
        self.input_dir = dir_input
        self.csv_files = sorted(glob.glob(os.path.join(dir_input, 'csv', "*.csv")))
        assert len(self.csv_files) > 0, "No csv file in {}".format(self.input_dir)
        self.dir_drawing = os.path.join(dir_input, "drawing")

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
        print("Loading CSV files")

        for i, f in enumerate(tqdm(self.csv_files)):
            class_name = filename_to_classname(f)
            src_name = os.path.splitext(os.path.basename(f))[0]
            id_class = CLASS_NAMES.index(class_name)
            df = pd.read_csv(f, index_col='key_id')
            df['id_class'] = id_class
            df['src_name'] = src_name
            df = df.sort_values("countrycode")
            index_train, index_valid = self._split_kfold(np.array(df.index.tolist()), idx_kfold, n_split)
            df_train_local = df.loc[index_train]
            df_valid_local = df.loc[index_valid]
            del df
            df_train.append(df_train_local)
            df_valid.append(df_valid_local)
            break

        df_train = pd.concat(df_train)
        df_valid = pd.concat(df_valid)

        dataset_train = DoodleDataset(df_train, self.dir_drawing, shape, transform=transforms.Compose([_ToTensor(), _Normalize()]))
        dataset_valid = DoodleDataset(df_valid, self.dir_drawing, shape, transform=transforms.Compose([_ToTensor(), _Normalize()]))

        return dataset_train, dataset_valid


class DoodleDataset(Dataset):

    def __init__(self, dataframe, dir_drawing, shape, transform=None):
        self.dataframe = dataframe
        self.dir_drawing = dir_drawing
        self.shape = shape
        self.transform = transform

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        npy = os.path.join(self.dir_drawing, row['src_name'], "{}.npy".format(row.name))
        image = draw_image(np.load(npy), shape=self.shape)
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


class _Normalize(object):
    def __call__(self, sample):
        id_class, image = sample['id_class'], sample['image']
        image = image.float() / 255
        return {'id_class': id_class, 'image': image}
