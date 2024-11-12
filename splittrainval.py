import os
import cv2
import numpy as np
import math
import random

f_read = open("bsds_pascal_train_pair.lst", "r")
f_train = open("bsds_pascal_train_pair_trainset.lst", "w")
f_val = open("bsds_pascal_train_pair_valset.lst", "w")


contents = f_read.readlines()
val_contents = []
train_contents = []

random.shuffle(contents)
count = 0
length = len(contents)
for content in contents:
    if count < length * 0.3:
        val_contents.append(content)
    else:
        train_contents.append(content)
    count = count + 1

f_val.writelines(val_contents)
f_train.writelines(train_contents)
f_val.close()
f_train.close()

