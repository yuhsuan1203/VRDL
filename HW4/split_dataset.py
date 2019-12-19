import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import glob
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco = COCO("pascal_train.json") # load training annotations

random_arr = np.random.choice(np.arange(0, 2), size=len(coco.imgs), p=[0.1, 0.9])
#print("There will be {} images in the validation set".format(len(random_arr)))

train_set = {}
valid_set = {}
train_set['annotations'] = []
train_set['images'] = []
train_set['categories'] = []
valid_set['annotations'] = []
valid_set['images'] = []
valid_set['categories'] = []

# categories
for i in coco.cats.keys():
    #print(coco.cats[i])
    train_set['categories'].append(coco.cats[i])
    valid_set['categories'].append(coco.cats[i])

# images
img_arr = list(coco.imgs.keys())
for i in range(len(img_arr)):
    if random_arr[i] == 1:
        train_set['images'].append(coco.imgs[img_arr[i]])
    else:
        valid_set['images'].append(coco.imgs[img_arr[i]])

# annotations
for i in range(len(img_arr)):
    if random_arr[i] == 1:
        for ann in coco.imgToAnns[img_arr[i]]:
            train_set['annotations'].append(ann)
    else:
        for ann in coco.imgToAnns[img_arr[i]]:
            valid_set['annotations'].append(ann)

if len(coco.anns) == (len(train_set['annotations']) + len(valid_set['annotations'])):
    print("Success")
    #print("Maybe right~~~")
else:
    print("Wrong!!!")

#print(len(train_set['annotations']))
#print(len(train_set['images']))
#print(len(train_set['categories']))
#print(len(valid_set['annotations']))
print("There are {} images in the validation set.".format(len(valid_set['images'])))
#print(len(valid_set['categories']))

with open("train_set.json", 'w') as outfile:
    json.dump(train_set, outfile)

with open("valid_set.json", 'w') as outfile:
    json.dump(valid_set, outfile)
