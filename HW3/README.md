# Street View House Numbers Detection using PyTorch-YOLOv3
A PyTorch implementation of YOLOv3

**Directory Tree**
```
HW3/
├── config/
│   ├── create_custom_model.sh
│   ├── custom.data
│   └── yolov3-custom.cfg
│
├── data/
│   └── custom/
│       ├── classes.names
│       ├── train.txt
│       └── valid.txt
│
├── utils/
│   ├── augmentations.py
│   ├── datasets.py
│   ├── logger.py
│   ├── parse_config.py
│   └── utils.py
│
├── weights/
│   └── download_weights.sh
│
├── benchmark.png
├── createlist.py
├── detect.py
├── label_annotation.py
├── mAP_45.txt
├── models.py
├── result.json
├── test.py
└── train.py
```

## Train on Custom Dataset

#### Custom model configuration ```config/yolov3-custom.cfg```
Execute the commands to create a custom model configuration file.
***10*** is the number of classes in the dataset.

```
$ cd config/                                
$ bash create_custom_model.sh 10 # create custom model configuration 'yolov3-custom.cfg'
```

#### Classes `data/custom/classes.names`
This file should have one row per class name. In this work, the file should have ten rows for ten different digits 0-9.

#### Image Folder `data/custom/images/`
Save the training and validation data.

#### Annotation Folder `data/custom/labels/`
##### **One file per image**
Each row in an annotation file defines one bounding box of the image, using the syntax `label_index x_center y_center bbox_width bbox_height`.

`label_index` should be **zero-indexed** and correspond to the row number of the class name in `data/custom/classes.names`. 

The coordinates (x_center, y_center, bbox_width, bbox_height) should be scaled to **[0, 1]** by dividing each value by the width and the height of the image respectively. 

The name of the annotation file should be the same as that of the image file in the image folder. For example, *1.txt* is for the image *1.png*.

```$ python label_annotation.py```

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt` , add paths to images used as train and validation data respectively.

``` $ python createlist.py```

#### [Optional]

Download pretrained weights:
    
    $ cd weights/
    $ bash download_weights.sh

#### Train
To train on the custom dataset run:

```
$ python -W ignore::UserWarning train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74
```

```--pretrained_weights weights/darknet53.conv.74 ``` : use a pretrained weight on ImageNet


## Inference
Make predictions on images:

    $ python detect.py [--image_folder folder_including_test_data] [--weights_path weights_path]
