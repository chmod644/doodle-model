from absl import flags

flags.DEFINE_string("input", "../chunk/train_simplified", "path to input directory", short_name="i")
flags.DEFINE_string("model", "../output/model", "path to model directory")
flags.DEFINE_string("log", "../output/log", "path to log directory")
flags.DEFINE_integer("worker", 4, "num workers of dataloader")
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_enum("archi", "resnet34",
                  enum_values=["resnet18", "resnet34", "resnet50",
                               "se_resnet50", "se_resnext50", "mobilenetv2", "original"],
                  help="model architecture")
flags.DEFINE_integer("img_height", 128, help="input image height")
flags.DEFINE_integer("img_width", 128, help="input image width")
flags.DEFINE_integer("batch_size", 256, help="batch size")
flags.DEFINE_integer("idx_kfold", 0, "index of k-fold validation")
flags.DEFINE_integer("kfold", 5, "num split to k-fold validation")
flags.DEFINE_integer('thickness', 1, "stroke thickness")
flags.DEFINE_bool('draw_first', False, "whether to draw image before resize")
flags.DEFINE_bool('white_background', False, "whether to draw strokes on white background")
