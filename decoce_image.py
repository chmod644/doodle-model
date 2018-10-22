#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os

import cv2
from absl import app, flags
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt

flags.DEFINE_string("input", "../input/train_simplified", "input directory", short_name="i")
flags.DEFINE_string("images", "../output/images", "images directory")

FLAGS = flags.FLAGS


def main(argv=None):
    files = list_files(FLAGS.input)
    files.sort()

    os.makedirs(FLAGS.images, exist_ok=True)

    for f in files:
        category = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        images = []
        for i, (key, row) in enumerate(df.iterrows()):
            strokes = literal_eval(row["drawing"])
            images.append(draw_image(strokes))
            if i == 12:
                break
        path_out = os.path.join(FLAGS.images, category + ".png")
        save_image(images, category, path_out)


def draw_image(strokes, shape=(256, 256, 3)):
    image = np.zeros(shape=shape, dtype=np.uint8)
    for stroke in strokes:
        stroke = np.transpose(np.array(stroke))
        stroke = np.int32([stroke])
        cv2.polylines(image, stroke, isClosed=False, color=(255, 255, 255))
    return image


def save_image(images, category, path_out):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(category)
    col = 6
    row = 2
    for i in range(col*row):
        ax = fig.add_subplot(row, col, i+1)
        ax.imshow(images[i])
    plt.savefig(path_out)


def list_files(dir):
    files = glob.glob(os.path.join(dir, "*.csv"))
    return files


if __name__ == '__main__':
    app.run(main)
