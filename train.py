#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

import sys
from pprint import pprint
from builtins import range

import numpy as np
from absl import app, flags
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import metrics
import util
from model import create_model, DEVICE
import config
from constant import *
from data import DatasetManager
from util_train import OptimizerManager

flags.DEFINE_integer("step", None, help="num of train steps")
flags.DEFINE_integer("epoch", None, help="num of train epoch")
flags.DEFINE_string("restart", None, help="path to restart checkpoint")

# Optimizer
flags.DEFINE_enum("optimizer", "adam", enum_values=["adam", "sgd"], help="optimizer")
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
        shape=(FLAGS.img_height, FLAGS.img_width, NUM_CHANNELS), mode='train',
        draw_first=FLAGS.draw_first, thickness=FLAGS.thickness, white_background=FLAGS.white_background, draw_contour=FLAGS.draw_contour, draw_contour_version=FLAGS.draw_contour_version)

    if FLAGS.debug:
        show_images(train_container)
        sys.exit()

    model = create_model(pretrained=FLAGS.pretrained, architecture=FLAGS.archi)
    if FLAGS.restart is not None:
        util.load_pth(model, FLAGS.restart)

    with open(os.path.join(FLAGS.model, "model.txt"), 'w') as f:
        print(model, file=f)

    optimizer = OptimizerManager(
        optimizer=FLAGS.optimizer, lr=FLAGS.lr, lr_decay=FLAGS.lr_decay, milestones=FLAGS.milestones,
        model=model, momentum=FLAGS.momentum)
    criterion = metrics.softmax_cross_entropy_with_logits()
    score_fn = metrics.map3()

    writer = SummaryWriter(log_dir=FLAGS.log)

    if FLAGS.step is not None:
        total_step = FLAGS.step
        total_epoch = None
    elif FLAGS.epoch is not None:
        total_step = np.iinfo(np.int32).max
        total_epoch = FLAGS.epoch
    else:
        raise AssertionError("step or epoch must be specified.")

    _ = train(model, optimizer, criterion, score_fn, train_container, valid_container,
              total_step=total_step, total_epoch=total_epoch, save_interval=FLAGS.save_interval, writer=writer)


def show_images(dataset_container):
    for dataset in dataset_container:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        for sample in loader:
            image = np.squeeze(sample['image'].numpy()).transpose(1, 2, 0)
            y = int(sample['y'][0])
            plt.imshow(1 - image)
            plt.title("{}. {}".format(y, ID_CATEGORY_DICT[y]))
            plt.show()


def train(model, optimizer, criterion, score_fn, train_container, valid_container, total_step, total_epoch,
          save_interval, writer):
    """

    :param model:
    :param optimizer:
    :param criterion:
    :param score_fn:
    :param total_step:
    :param total_epoch:
    :param train_container:
    :param valid_container:
    :param save_interval:
    :param writer:
    :return:
    """

    train_loader = train_container.batch_loader(batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.worker, epoch=total_epoch)
    valid_loader = valid_container.batch_loader(batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)

    global_step = 0
    num_subset = int(np.ceil(total_step / save_interval))
    for idx_subset in range(num_subset):
        sub_step = min(save_interval, total_step-idx_subset*save_interval)

        global_step, is_limit = train_subset(
            model, optimizer, criterion, train_loader, writer, global_step, sub_step=sub_step)

        if FLAGS.valid:
            validate(model, valid_loader, criterion, score_fn, writer, global_step)

        if is_limit:
            break

    return global_step


def train_subset(model, optimizer, criterion, train_loader, writer, global_step, sub_step):
    model.train()

    pbar = tqdm(train_loader, ascii=True, total=sub_step)

    try:
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

            # Write log for tensorboard
            if global_step % LOG_INTERVAL == 0:
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar('lr', optimizer.lr, global_step)

            # pbar update
            pbar.set_description("step:{}, train loss:{:6f}, lr:{:.2e}".format(global_step, loss, optimizer.lr))

    except RuntimeError as e:
        print("Total epoch reached to limit at {} step".format(global_step))
        torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILEFORMAT.format(global_step)))
        return global_step, True

    pbar.close()

    torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILEFORMAT.format(global_step)))

    return global_step, False


def validate(model, valid_loader, criterion, score_fn, writer, global_step):
    model.eval()
    merged_loss = 0
    merged_score = 0
    num_sample = 0
    pbar = tqdm(valid_loader, ascii=True, total=BATCH_VALID)
    for batch_idx, sample in enumerate(pbar):
        if batch_idx == BATCH_VALID:
            break
        ids_class, images = sample['y'].to(DEVICE), sample['image'].to(DEVICE)
        output = model(images)

        # Calc loss
        loss = criterion(output, ids_class)
        merged_loss += loss.item()
        average_loss = merged_loss / (batch_idx+1)

        # Calc score
        num_sample += ids_class.shape[0]
        score = score_fn(output, ids_class)
        merged_score += score.item()
        average_score = merged_score / num_sample

        pbar.set_description("step:{}, valid loss:{:6f}, valid score:{:6f}".format(
            global_step, average_loss, average_score))
    pbar.close()
    writer.add_scalar('val_loss', average_loss, global_step)
    writer.add_scalar('val_score', average_score, global_step)


if __name__ == '__main__':
    app.run(main)
