# coding=utf-8
# about captcha image
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 128
CHAR_SETS = 'abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
CLASSES_NUM = len(CHAR_SETS)

# 四位验证码
CHARS_NUM = 4

# for train
RECORD_DIR = './data'
TRAIN_FILE = 'train.tfrecords'
VALID_FILE = 'valid.tfrecords'
