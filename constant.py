import category

ID_CATEGORY_DICT = category.categories
CATEGORY_ID_DICT = {v: k for k, v in category.categories.items()}
NUM_CATEGORY = len(ID_CATEGORY_DICT)
BASE_SIZE = 256
ORIG_HEIGHT = 256
ORIG_WIDTH = 256
IMG_HEIGHT = 98
IMG_WIDTH = 98
NUM_CHANNELS = 3
MODEL_FILEFORMAT = "{}.pth"

