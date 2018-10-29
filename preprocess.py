#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import pandas as pd
from absl import app, flags
from tqdm import tqdm
from util import *

flags.DEFINE_string("input", "../input/train_simplified", "path to input directory", short_name="i")
flags.DEFINE_string('extract', '../work/train_simplified', 'path to extracted input')

FLAGS = flags.FLAGS

def main(argv=None):
    csv_files = sorted(glob.glob(os.path.join(FLAGS.input, "*.csv")))

    os.makedirs(FLAGS.extract, exist_ok=True)

    for i, f in enumerate(csv_files):
        print("Loading {}/{} {}".format(i+1, len(csv_files), f))
        df = load_csv(f)
        path_npy = os.path.join(FLAGS.extract, 'drawing')
        path_csv = os.path.join(FLAGS.extract, os.path.basename(f))
        save_csv_and_drawing(df, path_csv, path_npy)


def load_csv(csv_file):
    df = pd.read_csv(csv_file, index_col='key_id')
    df['drawing'] = df['drawing'].map(parse_drawing)
    return df


def save_csv_and_drawing(df, path_csv, path_npy):
    os.makedirs(path_npy, exist_ok=True)
    print("Saving drawing")
    for index, row in tqdm(df.iterrows(), total=len(df.index)):
        drawing = row['drawing']
        np.save(os.path.join(path_npy, "{}.npy".format(index)), drawing)

    df = df.drop('drawing', axis=1)
    print("Saving csv")
    df.to_csv(path_csv)


if __name__ == '__main__':
    app.run(main)