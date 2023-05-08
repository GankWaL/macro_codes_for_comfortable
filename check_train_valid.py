import glob
from tqdm import tqdm
import random
import os
import cv2
train_path = '/home/fssv3/wjh/cfg/FSNET_320_NIPA_sungjun/480K_dataset_train.txt'
valid_path = '/home/fssv3/wjh/cfg/FSNET_320_NIPA_sungjun/480K_dataset_valid.txt'

def load_txt(path):
    with open(path, "r") as f:
        return f.read().splitlines()
    
if __name__ == '__main__':
    train = load_txt(train_path)
    valid = load_txt(valid_path)
    v_ = []
    with open('/home/fssv3/wjh/dataset/ThinkI/OD/path_list/same_path_list.txt', 'w') as f:
        for v in tqdm(valid):
            if v in train:
                v_.append(v)
                f.write('\n'.join(v_))