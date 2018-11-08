#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extension package
"""
import numpy as np

def change_learning_rate(optimizer, lr):
    """
    method: change learning rate
    args 0: optimizer
    args 1: learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def float_to_color_1280(channel, ratio):
    ratio = np.maximum(0.0, np.minimum(ratio, 1.0))
    if channel != 3:
        return int(ratio * 255)
    value = int(ratio * 1279)
    if value < 256:
        return (255, value, 0)
    elif value < 512:
        return (511 - value, 255, 0)
    elif value < 768:
        return (0, 255, value - 512)
    elif value < 1024:
        return (0, 1023 - value, 255)
    else:
        return (value - 1024, value - 1024, 255)

def float_to_color_1024(channel, ratio):
    ratio = np.maximum(0.0, np.minimum(ratio, 1.0))
    if channel != 3:
        return int(ratio * 255)
    value = int(ratio * 1023)
    if value < 256:
        return (255, value, 0)
    elif value < 512:
        return (511 - value, 255, 0)
    elif value < 768:
        return (0, 255, value - 512)
    else:
        return (0, 1023 - value, 255)

def float_to_color_halftone(channel, ratio):
    ratio = np.maximum(0.0, np.minimum(ratio, 1.0))
    if channel != 3:
        return int(ratio * 127 + 128)
    value = int(ratio * 511)
    if value < 128:
        return (255, 128 + value, 128)
    elif value < 256:
        return (383 - value, 255, 128)
    elif value < 384:
        return (128, 255, value - 128)
    else:
        return (128, 639 - value, 255)
