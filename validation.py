#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from absl import app, flags
from torch.utils.data import DataLoader
from tqdm import tqdm

from constant import *
import config
from data import QuickDrawDataset, DatasetManager
from model import create_model, DEVICE
from util import load_pth

FLAGS = flags.FLAGS


def main(argv=None):

    model = create_model(pretrained=False, architecture=FLAGS.archi, is_train=False)
    model = load_pth(model, FLAGS.model)

    dataset_manager = DatasetManager(FLAGS.input)
    _, valid_dataset_container = dataset_manager.gen_train_and_valid(
        idx_kfold=FLAGS.idx_kfold, kfold=FLAGS.kfold,
        # 以下Datasetクラスに渡す引数
        shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), mode='train',
        draw_first=FLAGS.draw_first, thickness=FLAGS.thickness)

    total_correct = 0
    num_sample = 0
    arr_score_base = np.asarray([1.0, 1.0 / 2.0, 1.0 / 3.0])
    try:
        for i, dataset in enumerate(valid_dataset_container):
            print("Validating {}/{}".format(i+1, len(valid_dataset_container.csv_files)))
            num_sample += len(dataset)
            train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
            with torch.no_grad():
                for batch_idx, sample in enumerate(tqdm(train_loader, ascii=True)):
                    images, labels = sample['image'].to(DEVICE), sample['y'].to(DEVICE)
                    logits = model.forward(images)
                    val_logits_top_k, idx_logits_top_k = torch.topk(logits, 3)
                    idx_logits_top_k = idx_logits_top_k.t()
                    correct = idx_logits_top_k.eq(labels.view(1, -1).expand_as(idx_logits_top_k))
                    cur_pred = torch.sum(correct, dim=1).cpu().data.numpy()
                    cur_score = np.sum(np.multiply(cur_pred, arr_score_base))
                    total_correct += cur_score
                print("Validation score in 1~{} is {}".format(i+1, total_correct / num_sample))
    except StopIteration:
        pass

    val_score = total_correct / num_sample

    print("Validation score is {}".format(val_score))


if __name__ == '__main__':
    app.run(main)

