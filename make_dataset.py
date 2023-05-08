from natsort import natsorted
import glob
import os
import cv2
from tqdm import tqdm

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + dir)
    return dir

def make_dataset(path):
    # plist = glob.glob(path)
    # plist.sort()
    # for p in tqdm(plist):
    #     path_path = p + '\*'
    vlist = glob.glob(path)
    vlist.sort()
    # out_data_path = p.replace('DSM', 'DSM_frames')

    for videos in tqdm(vlist):
        if os.path.exists(videos):
            out_data_path = videos.replace('DSM', 'DSM_frames')
            out_data_path_ = out_data_path.split(".")[0]
            createFolder(out_data_path_)
        print(videos)
        file_name = videos.split("\\")[-1].split(".")[0]
        
        video = cv2.VideoCapture(videos)

        if not video.isOpened():
            print("Error opening video!")
        try:
            count = 0
            while video.isOpened():
                length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                # if (video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT)):
                #     video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ret, image = video.read()
                copy_image = image.copy()

                if (int(video.get(1)) % 30 == 0):
                    crop_image = copy_image[40:680, 640:1280]
                    print(f"Saved frame number: {int(video.get(1))}")
                    cv2.imwrite(out_data_path_ + '\\' + file_name + "_frame_%d.jpg" %count, crop_image)
                    with open(out_data_path_ + '\\' + file_name + "_frame_%d.txt" %count, "w", encoding='UTF-8') as f:
                        f.write(' ')
                    # image = cv2.flip(image, -1)
                    # cv2.imwrite(save_data_path + file_name + "_flip_frame_%d.jpg" %count, image)
                    print(out_data_path_ + '\\' + file_name + "_frame_%d.jpg" %count)
                    print(out_data_path_ + '\\' + file_name + "_frame_%d.txt" %count)
                    count += 1

            video.release()
        except:
            continue

if __name__ == '__main__':
    path = "D:\\test\\DSM\\think-i\\face_error\\*.avi"
    make_dataset(path)
