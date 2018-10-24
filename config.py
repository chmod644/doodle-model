import classes
from absl import flags

CLASS_NAMES = classes.class_names
NUM_CLASSES = len(CLASS_NAMES)
ORIG_HEIGHT = 256
ORIG_WIDTH = 256
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3

flags.DEFINE_string("input", "../input/train_simplified", "input directory", short_name="i")
flags.DEFINE_integer("idx_kfold", 0, "index of k-fold validation")
flags.DEFINE_integer("n_split", 5, "num split to k-fold validation")
