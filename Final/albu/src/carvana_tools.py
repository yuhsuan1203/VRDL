import cv2
import os
import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial
import json


def folds_mean(prob_file, roots, submissions_dir):
    ims = []
    #merge_fold = {'fma_s88': [1, 2, 4]} #, 'f04a_clahe_rmsprop': 2, 'fma_s44': 0, 'f04a_clahe': 0}
    for fold in range(5):
        for r in roots:
            #config_name = r.split('/')[-1]
            #print(config_name)
            #if fold not in merge_fold[config_name]:
                #print("Skip " + config_name + " fold {}".format(fold))
                #continue
            prob_path = os.path.join(r, 'fold{}_'.format(fold) + prob_file)
            im = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
    mean = (np.mean(ims, axis=0)).astype(np.uint8)
    cv2.imwrite(os.path.join(submissions_dir, 'albu27.09', prob_file), mean)


def parallel_mean():
    with open(os.path.join('..', '..', 'config', 'config_predict.json'), 'r') as f:
        config = json.load(f)
        submissions_dir = config['submissions_dir']
    root = r'../results'
    roots = [
        os.path.join(root, 'fma_s88'),
        #os.path.join(root, 'fma_s44'),
        #os.path.join(root, 'fma_noseed'),
        #os.path.join(root, 'f04a_clahe'),
        #os.path.join(root, 'f04a_s44'),
        #os.path.join(root, 'f04a_clahe_rmsprop')
    ]
    #print(roots)
    prob_files = os.listdir(roots[0])
    #print(len(prob_files))
    #print(prob_files[0])
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    print("unfolded {}".format(len(unfolded)))
    f = partial(folds_mean, roots=roots, submissions_dir=submissions_dir)
    os.makedirs(os.path.join(submissions_dir, 'albu27.09'), exist_ok=True)
    with Pool() as pool:
        pool.map(f, unfolded)

if __name__ == "__main__":
    freeze_support()
    parallel_mean()
