import glob
from tqdm import tqdm
import random
import os
import cv2
path = 'C:\\Users\\FS\\Desktop\\JHW\\think-i_K390_test\\test\\path_list\\New_phone_path.txt'

def load_txt(path):
    with open(path, "r") as f:
        return f.read().splitlines()
    
if __name__ == '__main__':
    files = load_txt(path)
    names = []
    count = {}
    for file in files:
        file_name = file.split('\\')[-1]
        names.append(file_name)
    
    for name in names:
        try: count[name] += 1
        except: count[name] = 1
    
    for key, value in count.items():
        if value == 2:
            print(key)
        
        