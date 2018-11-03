import glob
import os
import torch

from constant import *
from ast import literal_eval

import numpy as np



def get_submitname_dict():
    def _category_to_submitname(cat):
        return cat.replace(" ", "_")

    submitname_dict = {k: _category_to_submitname(v) for k, v in ID_CATEGORY_DICT.items()}
    return submitname_dict


def parse_drawing(str_drawing):
    _arrays = [np.array(x, dtype=np.uint8) for x in literal_eval(str_drawing)]
    arrays = np.zeros(shape=len(_arrays), dtype=object)
    for i, x in enumerate(arrays):
        arrays[i] = _arrays[i]
    return arrays


def reduce_sample(samples, max_element):
    """
    kfoldに分割

    :param samples:
    :param max_element:
    :return:
    """
    if len(samples) < max_element:
        print("Warning: len(samples)={} < max_element={}".format(len(samples), max_element))
        return samples
    indicies = np.linspace(0, (len(samples))-1, num=max_element, dtype=int)
    return samples[indicies]


def split_rotate(samples, kfold):
    """
    kfoldに分割

    :param samples:
    :param kfold:
    :return:
    """
    assert len(samples) > kfold
    num_sample = len(samples)
    list_samples = []
    for i in range(kfold):
        indicies = np.arange(i, num_sample, kfold)
        list_samples.append(samples[indicies])
    return list_samples


def split_kfold(samples, idx_kfold, kfold):
    """
    kfoldでtrain, sampleに分割する

    :param samples:
    :param idx_kfold:
    :param kfold:
    :return:
    """
    assert 0 <= idx_kfold < kfold
    assert len(samples) >= kfold
    num_sample = len(samples)
    indexes = np.arange(num_sample)
    splitted_indexes = np.array_split(indexes, kfold)
    valid_indexes = splitted_indexes.pop(kfold - idx_kfold -1)
    train_indexes = splitted_indexes
    if isinstance(train_indexes[0], np.ndarray):
        train_indexes = np.concatenate(train_indexes)
    samples_train = samples[train_indexes]
    samples_valid = samples[valid_indexes]
    return samples_train, samples_valid


def latest_pth(dirname):
    def _get_pth_step(filename):
        return int(os.path.splitext(os.path.basename(filename))[0])
    path_pth = sorted(glob.glob(os.path.join(dirname, '*.pth')), key=_get_pth_step)[-1]
    return path_pth


def load_pth(model, path):
    if os.path.isfile(path):
        path_pth = path
    elif os.path.isdir(path):
        path_pth = latest_pth(path)
    else:
        raise ValueError()
    print("Load from {}".format(path_pth))
    model.load_state_dict(torch.load(path_pth))
    return model


