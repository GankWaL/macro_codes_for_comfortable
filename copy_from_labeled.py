import glob
import plistlib
import shutil
import os.path
import random
from tqdm import tqdm

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + dir)
    return dir

def load_image(path):
    with open(path, "r") as f:
        return f.read().splitlines()
            
            
if __name__ == '__main__':
    path_labeled = 'C:\\Users\\FS\Desktop\\JHW\\think-i_K390_test\\test\\path_list\\New_phone_labeled_path.txt'
    path = 'C:\\Users\\FS\Desktop\\JHW\\think-i_K390_test\\test\\path_list\\New_phone_path.txt'
    images_labeled = load_image(path_labeled)
    out_path = load_image(path)
    for image in tqdm(images_labeled):
        for out_image in out_path:
            image_name = image.split('\\')[-1]
            out_image_name = out_image.split('\\')[-1]
            txt = image.split('.')[0] + '.txt'
            if image_name == out_image_name:
                # createFolder(out_path)
                # shutil.copy(image, out_image)
                # print("copy image")
                shutil.copy(txt, out_image.replace('.jpg', '.txt'))
                print("copy txt")
                print("------------------------------------------")