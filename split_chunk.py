#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import shutil

import pandas as pd
from absl import app, flags
from tqdm import tqdm
from util import *

flags.DEFINE_string("input", "../input/train_simplified", "path to csv directory", short_name="i")
flags.DEFINE_string('output', '../chunk/train_simplified', 'path to chunk csv directory')
flags.DEFINE_integer("num", 100, "number of chunk")
flags.DEFINE_integer("max_element", None, "max number of elements in each category")
flags.DEFINE_bool("debug", False, "debug mode. read only 300 rows from each category")

FLAGS = flags.FLAGS

PREFIX = 'train_k'


def main(argv=None):
    files_csv = sorted(glob.glob(os.path.join(FLAGS.input, '*.csv')))

    print('Create chunk csv files')
    root_csv = os.path.join(FLAGS.output)
    os.makedirs(root_csv, exist_ok=True)

    for i, f in enumerate(tqdm(files_csv, ascii=True)):
        if FLAGS.debug:
            nrows = 300
        else:
            nrows = None
        df = pd.read_csv(f, index_col='key_id', nrows=nrows)

        category_id_dict = {v: k for k, v in category.categories.items()}

        df['num_strokes'] = df['drawing'].map(lambda x: len(literal_eval(x)))
        df['y'] = df['word'].map(lambda x: category_id_dict[x])
        df_sorted = df.sort_values(by=['countrycode', 'recognized', 'num_strokes'])
        df_sorted = df_sorted.drop(columns='num_strokes')

        indicies = df_sorted.index.values

        list_indicies = split_rotate(indicies, FLAGS.num)
        for j, idxs in enumerate(list_indicies):

            if FLAGS.max_element is not None:
                max_element_chunk = FLAGS.max_element / FLAGS.num
                idxs = reduce_sample(idxs, max_element_chunk)

            df_split = df_sorted.loc[idxs]
            if i == 0:
                df_split.to_csv(os.path.join(root_csv, PREFIX + str(j) + '.csv'))
            else:
                df_split.to_csv(os.path.join(root_csv, PREFIX + str(j) + '.csv'), mode='a', header=False)


if __name__ == '__main__':
    app.run(main)

