import os
import numpy as np
import shutil
import random

# Works when all the patches are in the train folder and some need to be moved to valid
data_path = '/project/DSone/as3ek/data/patches/1000/gannorm_seem_cinn/'
train_perct = 90
source = 'train/Normal/'
target = 'valid/Normal/'

patch_list = os.listdir(data_path + source)
unq_biopsy = np.unique([x.split('__')[0] for x in patch_list])

unq_biopsy

patch_target_map = {}
for i, patch in enumerate(patch_list):
    if patch.split('__')[0] not in patch_target_map:
        if random.randint(1, 100) < train_perct:
            patch_target_map[patch.split('__')[0]] = 'dont_move'
        else:
            patch_target_map[patch.split('__')[0]] = 'move'
    
    if i % 1000 == 0:
        print(i)
    
    if patch_target_map[patch.split('__')[0]] == 'move':
        shutil.move(data_path + source + patch, data_path + target)
    else:
        continue


