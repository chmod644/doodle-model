#!/usr/bin/env python
# -*- coding: utf-8 -*-

from absl import app, flags
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import QuickDrawDataset
from model import create_model, DEVICE
from torch.autograd import Variable
import glob
import os
from constant import *

from util import load_pth, get_submitname_dict

flags.DEFINE_string('submission', '../output/submission.csv', 'path to submission file')
flags.DEFINE_string('inference', '../output/inference.pickle', 'path to inference pickle file')

FLAGS = flags.FLAGS


def main(argv=None):
    model = create_model(pretrained=False, architecture=FLAGS.archi, is_train=False)
    model = load_pth(model, FLAGS.model)

    path_test = FLAGS.input
    dataset = QuickDrawDataset(path_test, shape=(FLAGS.img_height, FLAGS.img_width, NUM_CHANNELS), mode='test',
                               draw_first=FLAGS.draw_first, thickness=FLAGS.thickness,
                               white_background=FLAGS.white_background)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    categories_dict = get_submitname_dict()
    softmax = torch.nn.Softmax()

    inference_result_list = []
    submit_list = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, ascii=True)):
            images, key_id = sample['image'].to(DEVICE), sample['key_id']
            logits = model.forward(images)
            softmax_logits = softmax(logits)
            logits_top_k, idx_logits_top_k = torch.topk(logits, 3)
            for cur_key_id, cur_idx_logits_top_k, cur_softmax_logits in zip(key_id, idx_logits_top_k, softmax_logits):
                key_id = cur_key_id.item()
                inference_result_list.append([key_id] + cur_softmax_logits.cpu().data.numpy().tolist())
                submit_list.append({"key_id": key_id,
                                    "word": "{} {} {}".format(categories_dict[cur_idx_logits_top_k[0].item()],
                                                              categories_dict[cur_idx_logits_top_k[1].item()],
                                                              categories_dict[cur_idx_logits_top_k[2].item()])})

    os.makedirs(os.path.dirname(FLAGS.submission), exist_ok=True)
    export_df = pd.DataFrame(submit_list)
    export_df.to_csv(FLAGS.submission, index=False)

    os.makedirs(os.path.dirname(FLAGS.inference), exist_ok=True)
    infres_df = pd.DataFrame(inference_result_list)
    infres_df.to_pickle(FLAGS.inference)


if __name__ == '__main__':
    app.run(main)
