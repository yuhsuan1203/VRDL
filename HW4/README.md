# Instance Segmentation - YOLACT v1.1
A fully convolutional model for instance segmentation.

## Split **pascal_train.json** for training and validation
```
python split_dataset.py
```
generate train_set.json and valid_set.json

## Model Architecture
![](https://i.imgur.com/JyVeR0x.png)

## Model Configuration
Set up the model configuration in ```data/config.py```
In this work, **yolact_resnet101_pascal_config** is used.

## Optional
**In root directory:**   ```mkdir weights``` and store any pre-trained weights into ```weights/```

## Training
```
python train.py --config=yolact_resnet101_pascal_config \
--save_interval=7500 --validation_epoch=5
```

## Evaluation
```
python eval_submission.py --config=yolact_resnet101_pascal_config \ 
--trained_model=[weight] --score_threshold=0.15 --top_k=15 \ 
--images=[input image folder]:[output image folder]
```
