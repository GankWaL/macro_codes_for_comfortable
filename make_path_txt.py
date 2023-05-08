import glob
import os.path
from tqdm import tqdm

def make_txt_for_path_list(path):
    plist = glob.glob(path)
    path_list = [p for p in plist if p.endswith(".jpg")]
    # print(plist)
    # print(path_list)
    # test = images_path.split('*')[0]
    path_list.sort()
    with open('D:\\test\\DSM_frames\\path_list\\sunvisor_path.txt', "w", encoding='UTF-8' ) as f:
        f.write('\n'.join(path_list))
        
if __name__ == '__main__':
    path = "D:\\test\\DSM_frames\\think-i\\limit_test\\sunvisor\\*\\*.jpg"
    make_txt_for_path_list(path)