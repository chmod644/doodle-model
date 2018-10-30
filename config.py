import classes
from absl import flags

CLASS_NAMES = classes.class_names
NUM_CLASSES = len(CLASS_NAMES)
ORIG_HEIGHT = 256
ORIG_WIDTH = 256
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3
MODEL_FILENAME = "checkpoint.pt"

flags.DEFINE_string("input", "../work/train_simplified", "path to input directory", short_name="i")
flags.DEFINE_string("model", "../output/model", "path to model directory")
flags.DEFINE_integer("worker", 4, "num workers of dataloader")
flags.DEFINE_bool("debug", False, "debug mode")
