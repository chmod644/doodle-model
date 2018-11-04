#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

import sys
from pprint import pprint

import numpy as np
from absl import app, flags
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import create_model, DEVICE
import config
from constant import *
from data import DatasetManager
from util_train import OptimizerManager

flags.DEFINE_integer("epochs", 20, help="num of train epochs")
flags.DEFINE_enum("optim", 'adam', ['adam', 'sgd'], help="optimizer")
flags.DEFINE_float("lr", 0.0001, help="learning rate")
flags.DEFINE_list("milestones", None, help="decay milestones of learning rate")
flags.DEFINE_float("lr_decay", 1.0, help="decay factor for learning rate")
flags.DEFINE_float("momentum", 0.5, help="SGD momentum")
flags.DEFINE_integer("save_interval", 10000, "inteval of step to save checkpoint")
flags.DEFINE_bool('valid_subset', True, "whether to validate subset of dataset")
flags.DEFINE_bool('pretrained', False, "whether to use pretrained model")

FLAGS = flags.FLAGS


def main(argv=None):
    os.makedirs(FLAGS.model, exist_ok=True)

    with open(os.path.join(FLAGS.model, 'config.json'), 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f, indent=4, sort_keys=True)

    dataset_manager = DatasetManager(FLAGS.input)
    train_dataset_iter, valid_dataset_iter = dataset_manager.gen_train_and_valid(
        idx_kfold=FLAGS.idx_kfold, kfold=FLAGS.kfold,
        # 以下Datasetクラスに渡す引数
        shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), mode='train',
        draw_first=FLAGS.draw_first, thickness=FLAGS.thickness)

    if FLAGS.debug:
        # 画像表示テスト
        for train_dataset in train_dataset_iter:
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
            for elem in train_loader:
                image = np.squeeze(elem['image'].numpy()).transpose(1, 2, 0)
                y = int(elem['y'][0])
                plt.imshow(1 - image)
                plt.title("{}. {}".format(y, ID_CATEGORY_DICT[y]))
                plt.show()
        sys.exit()

    model = create_model(pretrained=FLAGS.pretrained, architecture=FLAGS.archi)

    print(model)
    with open(os.path.join(FLAGS.model, "model.txt"), 'w') as f:
        print(model, file=f)

    optimizer = OptimizerManager(
        optimizer=FLAGS.optim, lr=FLAGS.lr, lr_decay=FLAGS.lr_decay, milestones=FLAGS.milestones,
        model=model, momentum=FLAGS.momentum)
    criterion = torch.nn.NLLLoss()

    global_step = 0
    for global_epoch in range(FLAGS.epochs):
        global_step = train_epoch(global_step, global_epoch, model, optimizer, criterion,
                train_dataset_iter, valid_dataset_iter, save_interval=FLAGS.save_interval,
                valid_subset=FLAGS.valid_subset)


def train_epoch(global_step, global_epoch, model, optimizer, criterion,
                train_dataset_iter, valid_dataset_iter, save_interval=10000, valid_subset=True):
    """
    1epoch分の学習。学習が終わったらvalidation setのlossを出す

    :param global_step:
    :param global_epoch:
    :param model:
    :param optimizer:
    :param criterion:
    :param device:
    :param train_dataset_iter:
    :param valid_dataset_iter:
    :param save_interval:
    :param valid_subset:
    :return:
    """
    merged_loss = 0.
    for idx_subset, train_dataset in enumerate(train_dataset_iter):
        train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)

        local_loss, global_step = train(model, train_loader,
                                  optimizer, criterion, global_step, save_interval)

        merged_loss += local_loss
        print("Training {}/{} in epoch {} (step={}) loss:{:.6f}".format(
            idx_subset+1, len(train_dataset_iter), global_epoch+1, global_step+1, local_loss))
    averaged_loss = merged_loss / len(train_dataset_iter)
    print("Training epoch:{}, loss:{:.6f}".format(global_epoch, averaged_loss))

    merged_loss = 0.
    for idx_subset, valid_dataset in enumerate(valid_dataset_iter):
        valid_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
        local_loss = validate(model, valid_loader, criterion=criterion)
        if valid_subset:
            # validation set の一部だけ評価
            print("Validation epoch:{}, loss:{:.6f}".format(global_epoch+1, local_loss))
            return global_step
    merged_loss += local_loss
    averaged_loss = merged_loss / len(valid_dataset_iter)
    print("Validation epoch:{}, loss:{:.6f}".format(global_epoch+1, averaged_loss))

    return global_step


def train(model, train_loader, optimizer, criterion, global_step=0, save_interval=10000):
    model.train()
    merged_loss = 0.
    for batch_idx, sample in enumerate(tqdm(train_loader, ascii=True)):
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        optimizer.optimizer.zero_grad()
        output = model(images)
        output = torch.nn.functional.log_softmax(output, 1)
        loss = criterion(output, ids_class)
        loss.backward()
        merged_loss += loss.item()
        optimizer.step()
        global_step += 1
        if global_step % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILEFORMAT.format(global_step)))
    average_loss = merged_loss / len(train_loader)
    return average_loss, global_step


def validate(model, valid_loader, criterion):
    model.eval()
    merged_loss = 0
    for batch_idx, sample in enumerate(tqdm(valid_loader, ascii=True)):
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        output = model(images)
        output = torch.nn.functional.log_softmax(output)
        loss = criterion(output, ids_class)
        merged_loss += loss.item()
    average_loss = merged_loss / len(valid_loader)
    return average_loss


if __name__ == '__main__':
    app.run(main)
