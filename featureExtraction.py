import os, re
import numpy as np
from PIL import Image
from functools import reduce
import pickle

IMG_DIR = os.getcwd() + "/saved_img"

filelist, featurelist, targetlist = [], [], []
compiler = re.compile(r"label|[0-9]{1,2}") # produce tuple of 3 (label, label_num, draw_num)


for root, dirname, filenames in os.walk(IMG_DIR):

    for filename in filenames:
        filelist.append(filename)
        result = compiler.findall(filename)

        img = Image.open(IMG_DIR + "/" + filename)
        feature = np.asarray(img).reshape(1, 28, 28, 1)
        target = result[1]

        featurelist.append(feature)
        targetlist.append(target)

def concatArray(first, second):
    result = np.concatenate([first, second], axis=0)
    return result


features = reduce(concatArray, featurelist)
targets = np.array(targetlist)

print(features.shape, targets.shape)

for elements, name in ([features, "features"], [targets, "targets"]):
    with open("saved_model/%s.pkl" % (name), "wb") as file:
        pickle.dump(elements, file)