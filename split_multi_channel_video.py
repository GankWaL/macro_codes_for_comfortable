import argparse
import glob
import os.path
import ffmpeg
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="C:\\Users\\FS\\Desktop\\JHW\\think-i_K390_test\\test\\think-i\\*",
                        help="2채널 영상 데이터셋 폴더, 불러오기 영상 파일이 있는 복수의 서브폴더 만큼 /* 추가 ex) C:\\path\\to\\videos\\* , home/path/to/videos/*")
    return parser.parse_args()

def check_arguments_errors(args):
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + dir)
    return dir

def split_multi_channel_video(path):
    plist = glob.glob(path)
    plist.sort()
    for p in tqdm(plist):
        path_path = p + '\*'
        vlist = glob.glob(path_path)
        vlist.sort()
        out_path_dsm = p.replace('\\test\\', '\\test\\DSM\\')
        out_path_fcws = p.replace('\\test\\', '\\test\\FRONT\\')
        createFolder(out_path_dsm)
        createFolder(out_path_fcws)
        for v in tqdm(vlist):
            if v.endswith('.avi'):
                if not (os.path.exists(out_path_fcws + '\\' + v.split('\\')[-1]) or os.path.exists(out_path_dsm + '\\' + v.split('\\')[-1])):
                    stream = ffmpeg.input(v)
                    front = stream['v:0']
                    incabin = stream['v:1']
                    audio = stream['a']
                    try:
                        output_fcws =  ffmpeg.output(front, out_path_fcws + '\\' + v.split('\\')[-1], **{'vcodec':'h264'})
                        ffmpeg.run(output_fcws)
                    except:
                        continue
                    try:
                        output_dsm = ffmpeg.output(incabin, out_path_dsm + '\\' + v.split('\\')[-1], **{'vcodec':'h264'})
                        ffmpeg.run(output_dsm)
                    except:
                        continue  
                
if __name__ == '__main__':
    args = parser()
    # check_arguments_errors(args)
    path = args.input
    split_multi_channel_video(path)
