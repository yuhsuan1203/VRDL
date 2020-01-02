# VRDL_Final
## Code
 - `albu`:
     - `configs`: Contains the training config
     - `src`: 
         - `train.py`: Main function for training
         - `transforms.py`: Data augmentation functions
         - `pytorch_zoo/linknet.py`: LinkNet definition
         - `carvana_eval.py`: Main function for evaluation
         - `carvana_tools.py`: Functions used to generate the average of the predicted images
         - `eval.py`: Evaluator definition
         - `dataset`: Some utils for loading dataset
         - `utils.py`: Utils for parsing configs and csv
         - `generate_folds.py`: Generate data splitting file
     - `fma.csv`: One way to spilt dataset
     - `predict.sh`: Prediction script
     - `train.sh`: Training script
 - `asanakoy`:
     - `data_utils.py`: RLE encode & RLE to string function
     - `dataset.py`: Functions to load test data
 - `config`: Input/Output path configs
 - `generate_sub_final_ensemble.py`: Functions used to generate the final submission file
 - `requirements.txt`: Package requirements
 - `setup_env.sh`: Script for setting up the environment

## Reference
Carvana Image Masking Challenge Rank 1st

https://github.com/asanakoy/kaggle_carvana_segmentation

## Installation
```
# bash setup_env.sh
```

## Training
```
# cd albu
# bash train.sh
```

## Predict
```
# cd albu
# bash predict.sh
# cd ..
# source activate py35_albu
# python generate_sub_final_ensemble.py
# conda deactivate
```
