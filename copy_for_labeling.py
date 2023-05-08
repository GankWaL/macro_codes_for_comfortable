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
    path = 'D:\\test\\DSM_frames\\path_list\\sunvisor_path.txt'
    images = load_image(path)
    for image in tqdm(images):
        out_path = 'D:\\test\\DSM_frames\\think-i\\workspace\\sunvisor\\' + image.split('\\')[-1]
        txt = image.split('.')[0] + '.txt'
        if os.path.exists(image):
            # createFolder(out_path)
            shutil.copy(image, out_path)
            print("copy image")
            # shutil.copy(txt, out_path.replace('.jpg', '.txt'))
            print("copy txt")
            print("------------------------------------------")