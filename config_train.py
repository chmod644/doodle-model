from absl import flags
from config import *

flags.DEFINE_enum("resnet_type", "resnet34", enum_values=["resnet34", "resnet50"], help="model type")
flags.DEFINE_integer("epochs", 20, help="num of train epochs")
flags.DEFINE_float("lr", 0.01, help="learning rate")
flags.DEFINE_float("momentum", 0.5, help="SGD momentum")
flags.DEFINE_integer("batch_size", 20, help="batch size")
flags.DEFINE_integer("idx_kfold", 0, "index of k-fold validation")
flags.DEFINE_integer("n_split", 5, "num split to k-fold validation")
