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

import metrics
from model import create_model, DEVICE
import config
from constant import *
from data import DatasetManager
from util_train import OptimizerManager

flags.DEFINE_integer("step", None, help="num of train steps")
flags.DEFINE_integer("epoch", None, help="num of train epoch")

# Optimizer
flags.DEFINE_enum("optim", 'adam', ['adam', 'sgd'], help="optimizer")
flags.DEFINE_float("lr", 0.0001, help="learning rate")
flags.DEFINE_float("lr_decay", 1.0, help="decay factor for learning rate")
flags.DEFINE_list("milestones", None, help="decay milestones of learning rate")
flags.DEFINE_float("momentum", 0.5, help="SGD momentum")

# Checkpoint and log configuration
flags.DEFINE_integer("save_interval", 10000, "inteval of step to save checkpoint")
flags.DEFINE_bool('pretrained', False, "whether to use pretrained model")
flags.DEFINE_bool('valid', True, "whether to validate after save checkpoint")

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
        idx_kfold=FLAGS.idx_kfold, kfold=FLAGS.kfold, shuffle_train=True, shuffle_valid=True, verbose=False,
        # Parameters for DatasetClass
        shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), mode='train',
        draw_first=FLAGS.draw_first, thickness=FLAGS.thickness)

    if FLAGS.debug:
        show_images(train_container)
        sys.exit()

    model = create_model(pretrained=FLAGS.pretrained, architecture=FLAGS.archi)

    with open(os.path.join(FLAGS.model, "model.txt"), 'w') as f:
        print(model, file=f)

    optimizer = OptimizerManager(
        optimizer=FLAGS.optim, lr=FLAGS.lr, lr_decay=FLAGS.lr_decay, milestones=FLAGS.milestones,
        model=model, momentum=FLAGS.momentum)
    criterion = metrics.softmax_cross_entropy_with_logits()

    writer = SummaryWriter(log_dir=FLAGS.log)

    if FLAGS.step is not None:
        total_step = FLAGS.step
    elif FLAGS.epoch is not None:
        total_step = int(np.ceil(FLAGS.epoch * train_container.num_sample() / FLAGS.batch_size))
    else:
        raise AssertionError("step or epoch must be specified.")
    print("Total step is {}".format(total_step))

    _ = train(model, optimizer, criterion, train_container, valid_container, total_step=total_step,
              save_interval=FLAGS.save_interval, writer=writer)


def show_images(dataset_container):
    for dataset in dataset_container:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        for sample in loader:
            image = np.squeeze(sample['image'].numpy()).transpose(1, 2, 0)
            y = int(sample['y'][0])
            plt.imshow(1 - image)
            plt.title("{}. {}".format(y, ID_CATEGORY_DICT[y]))
            plt.show()


def train(model, optimizer, criterion, train_container, valid_container, total_step, save_interval=10000, writer=None):
    """

    :param model:
    :param optimizer:
    :param criterion:
    :param total_step:
    :param train_container:
    :param valid_container:
    :param save_interval:
    :return:
    """

    train_loader = train_container.batch_loader(batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.worker)
    valid_loader = valid_container.batch_loader(batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)

    global_step = 0
    num_subset = int(np.ceil(total_step / save_interval))
    for idx_subset, _ in enumerate(range(num_subset)):
        _, global_step = train_subset(model, optimizer, criterion, train_loader, writer, global_step,
                                      sub_step=save_interval)

        if FLAGS.valid:
            _ = validate(model, valid_loader, criterion, writer, global_step)

    return global_step


def train_subset(model, optimizer, criterion, train_loader, writer, global_step, sub_step):
    model.train()
    merged_loss = 0.

    pbar = tqdm(train_loader, ascii=True, total=sub_step)

    for batch_idx, sample in enumerate(pbar):
        if batch_idx == sub_step:
            break

        # Train batch
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, ids_class)
        loss.backward()
        optimizer.step()
        global_step += 1
        merged_loss += loss.item()

        # Write log for tensorboard
        if global_step % LOG_INTERVAL == 0:
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('lr', optimizer.lr, global_step)

        # pbar update
        pbar.set_description("step:{}, train loss:{:6f}, lr:{:.2e}".format(global_step, loss, optimizer.lr))

    pbar.close()

    torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILEFORMAT.format(global_step)))

    average_loss = merged_loss / sub_step
    return average_loss, global_step


def validate(model, valid_loader, criterion, writer, global_step):
    model.eval()
    merged_loss = 0
    pbar = tqdm(valid_loader, ascii=True, total=BATCH_VALID)
    for batch_idx, sample in enumerate(pbar):
        pbar.set_description("validation {}/{}".format(batch_idx, BATCH_VALID))
        if batch_idx == BATCH_VALID:
            break
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        output = model(images)
        loss = criterion(output, ids_class)
        merged_loss += loss.item()
    average_loss = merged_loss / BATCH_VALID
    pbar.clear()
    print("step:{}, valid loss:{:6f}".format(global_step, average_loss))
    writer.add_scalar('val_loss', average_loss, global_step)
    return average_loss


if __name__ == '__main__':
    app.run(main)
