#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import sys

import numpy as np
import torch
from absl import app, flags
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import create_resnet
from config_train import *
from data import DatasetGenerator

FLAGS = flags.FLAGS


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    merged_loss = 0
    for batch_idx, elem in enumerate(tqdm(train_loader)):
        ids_class, images = elem['id_class'].to(device), elem['image'].to(device)
        optimizer.zero_grad()
        output = model(images)
        output = torch.nn.functional.log_softmax(output, 1)
        loss = criterion(output, ids_class)
        loss.backward()
        merged_loss += loss.item()
        optimizer.step()
    merged_loss = merged_loss / len(train_loader)
    print("Training epoch:{}, loss:{:.6f}".format(epoch, merged_loss))


def validate(model, device, train_loader, criterion, epoch):
    model.train()
    merged_loss = 0
    for batch_idx, elem in enumerate(tqdm(train_loader)):
        ids_class, images = elem['id_class'].to(device), elem['image'].to(device)
        output = model(images)
        output = torch.nn.functional.log_softmax(output)
        loss = criterion(output, ids_class)
        merged_loss += loss.item()
    merged_loss = merged_loss / len(train_loader)
    print("Validation epoch:{}, loss:{:.6f}".format(epoch, merged_loss))


def main(argv=None):
    os.makedirs(FLAGS.model, exist_ok=True)

    dataset = DatasetGenerator(FLAGS.input)
    train_dataset, valid_dataset = dataset.gen_train_and_valid(
        shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), idx_kfold=FLAGS.idx_kfold, n_split=FLAGS.n_split)

    if FLAGS.debug:
        image = train_dataset[0]['image'].numpy().transpose(1, 2, 0)
        plt.imshow(image)
        plt.show()
        sys.exit()

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_resnet(pretrained=False, resnet_type=FLAGS.resnet_type)
    model.to(device)

    print(model)
    with open(os.path.join(FLAGS.model, "model.txt"), 'w') as f:
        print(model, file=f)

    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
    criterion = torch.nn.NLLLoss()

    for epoch in range(FLAGS.epochs):
        train(model, device, train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch)
        validate(model, device, valid_loader, criterion=criterion, epoch=epoch)
        torch.save(model.state_dict(), os.path.join(FLAGS.model, MODEL_FILENAME))


if __name__ == '__main__':
    app.run(main)
