#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

import sys
from pprint import pprint

import numpy as np
from absl import app, flags
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import create_model, DEVICE
import config
from constant import *
from data import DatasetManager
from util_train import OptimizerManager

flags.DEFINE_integer("step", 1000000, help="num of train steps")
flags.DEFINE_enum("optim", 'adam', ['adam', 'sgd'], help="optimizer")
flags.DEFINE_float("lr", 0.0001, help="learning rate")
flags.DEFINE_list("milestones", None, help="decay milestones of learning rate")
flags.DEFINE_float("lr_decay", 1.0, help="decay factor for learning rate")
flags.DEFINE_float("momentum", 0.5, help="SGD momentum")
flags.DEFINE_integer("save_interval", 10000, "inteval of step to save checkpoint")
flags.DEFINE_bool('valid_subset', True, "whether to validate subset of dataset")
flags.DEFINE_bool('pretrained', False, "whether to use pretrained model")

FLAGS = flags.FLAGS

LOG_INTERVAL = 1000
BATCH_VALID = 100


def main(argv=None):
    os.makedirs(FLAGS.model, exist_ok=True)
    os.makedirs(FLAGS.log, exist_ok=True)

    with open(os.path.join(FLAGS.model, 'config.json'), 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f, indent=4, sort_keys=True)

    dataset_manager = DatasetManager(FLAGS.input)
    train_container, valid_container = dataset_manager.gen_train_and_valid(
        idx_kfold=FLAGS.idx_kfold, kfold=FLAGS.kfold, shuffle_train=True, shuffle_valid=True,
        # 以下Datasetクラスに渡す引数
        shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), mode='train',
        draw_first=FLAGS.draw_first, thickness=FLAGS.thickness)

    # 画像表示テスト
    if FLAGS.debug:
        show_images(train_container)

    model = create_model(pretrained=FLAGS.pretrained, architecture=FLAGS.archi)

    with open(os.path.join(FLAGS.model, "model.txt"), 'w') as f:
        print(model, file=f)

    optimizer = OptimizerManager(
        optimizer=FLAGS.optim, lr=FLAGS.lr, lr_decay=FLAGS.lr_decay, milestones=FLAGS.milestones,
        model=model, momentum=FLAGS.momentum)
    criterion = torch.nn.NLLLoss()

    writer = SummaryWriter(log_dir=FLAGS.log)

    _ = train(model, optimizer, criterion, train_container, valid_container, total_step=FLAGS.step,
              save_interval=FLAGS.save_interval, writer=writer)


def show_images(dataset_container):
    for dataset_dataset in dataset_container:
        loader = DataLoader(dataset_dataset, batch_size=1, shuffle=True, num_workers=4)
        for sample in loader:
            image = np.squeeze(sample['image'].numpy()).transpose(1, 2, 0)
            y = int(sample['y'][0])
            plt.imshow(1 - image)
            plt.title("{}. {}".format(y, ID_CATEGORY_DICT[y]))
            plt.show()
    sys.exit()


def train(model, optimizer, criterion, train_container, valid_container, total_step, save_interval=10000, writer=None):
    """
    1epoch分の学習。学習が終わったらvalidation setのlossを出す

    :param model:
    :param optimizer:
    :param criterion:
    :param total_step:
    :param train_container:
    :param valid_container:
    :param save_interval:
    :return:
    """
    global_step = 0

    num_subset = int(np.ceil(total_step / save_interval))
    for idx_subset, _ in enumerate(range(num_subset)):
        local_loss, global_step = train_subset(model, optimizer, criterion, train_container, global_step,
                                               sub_step=save_interval, writer=writer)
        print("Training loss:{:.6f} (step={})".format(local_loss, global_step + 1))

        # local_loss = validate(model, valid_container, criterion=criterion)
        # print("Validation loss:{:.6f} (step={})".format(global_step+1, local_loss))

    return global_step


def train_subset(model, optimizer, criterion, train_container, global_step=0, sub_step=10000, writer=None):
    model.train()
    merged_loss = 0.

    pbar = tqdm(
        train_container.batch(batch_size=FLAGS.batch_size, shuffle=True, num_workers=4),
        ascii=True, total=sub_step)

    for batch_idx, sample in enumerate(pbar):
        if batch_idx == sub_step:
            break

        # Train sample
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        output = torch.nn.functional.log_softmax(output, 1)
        loss = criterion(output, ids_class)
        loss.backward()
        optimizer.step()
        merged_loss += loss.item()

        # Write log for tensorboard
        if global_step % LOG_INTERVAL == 0:
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('lr', optimizer.lr, global_step)

        # pbar update
        pbar.set_description("step:{}, loss:{:6f}, lr:{:.2e}".format(global_step, loss, optimizer.lr))
        pbar.update()

        global_step += 1

    torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILEFORMAT.format(global_step)))

    average_loss = merged_loss / sub_step
    return average_loss, global_step


def validate(model, valid_container, criterion):
    model.eval()
    merged_loss = 0
    pbar = tqdm(valid_container.batch(batch_size=FLAGS.batch_size, shuffle=True, num_workers=1),
                ascii=True, total=BATCH_VALID)
    for batch_idx, sample in enumerate(pbar):
        if batch_idx == BATCH_VALID:
            break
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        output = model(images)
        output = torch.nn.functional.log_softmax(output)
        loss = criterion(output, ids_class)
        merged_loss += loss.item()
        pbar.update()
    average_loss = merged_loss / BATCH_VALID
    return average_loss


if __name__ == '__main__':
    app.run(main)
