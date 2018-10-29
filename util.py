import os
from ast import literal_eval

import numpy as np


def filename_to_classname(filename):
    return os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")


def parse_drawing(str_drawing):
    _arrays = [np.array(x, dtype=np.uint8) for x in literal_eval(str_drawing)]
    arrays = np.zeros(shape=len(_arrays), dtype=object)
    for i, x in enumerate(arrays):
        arrays[i] = _arrays[i]
    return arrays

