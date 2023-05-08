
import numpy as np
import glob
import math
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
import cv2
from tqdm import tqdm

listOfbackground = glob.glob("/home/fssv3/wjh/augmentation/background/*")

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: 해당 경로에 폴더를 만들 수 없음 ' + dir)

def load_images(path):
    with open(path, "r") as f:
        return f.read().splitlines()
        
def getAlphaImage(img1Path, img2Path):
    img = cv2.imread(img1Path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_offset = y_offset = 0
    # threshold input image using otsu thresholding as mask and refine with morphology
    ret, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # put mask into alpha channel of image
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # print("mask shape",mask.shape)
    result[:, :, 3] = mask

    s_img = result#cv2.imread('retina_masked.png', -1)

    l_img = cv2.imread(img2Path)
    l_img = cv2.resize(l_img, (s_img.shape[1], s_img.shape[0]))
    #print(img.shape, l_img.shape, s_img.shape)

    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                alpha_l * l_img[y1:y2, x1:x2, c])
    cv2.imwrite(img1Path, l_img)

def saveAug(image_aug, bbs_aug, originalPath, augType, blend = 0 ):
    # pathImgAug = originalPath.replace(".jpg", "_" + augType + str(num) + ".jpg")
    pathImgAug = originalPath.replace(".jpg", "_" + augType+ ".jpg").replace('Resize', 'Resize/aug')
    createFolder(pathImgAug.strip(pathImgAug.split('/')[-1]))
    imageio.imwrite(pathImgAug, image_aug)
    if blend == 1:
        getAlphaImage(pathImgAug, listOfbackground[random.randint(0, len(listOfbackground))])
    # pathBoxAug = originalPath.replace(".jpg", "_" + augType + str(num) + ".txt")
    pathBoxAug = originalPath.replace(".jpg", "_" + augType+ ".txt").replace('Resize', 'Resize/aug')
    with open(pathBoxAug, 'w') as f:
        for i in range(len(bbs_aug.bounding_boxes)):
            augBoxCordinate = bbs_aug.bounding_boxes[i]
            x_center,y_center,width,height = imgaug2Yolo((image_aug.shape[1], image_aug.shape[0]), (augBoxCordinate.x1, augBoxCordinate.x2, augBoxCordinate.y1, augBoxCordinate.y2) )
            if width > 0 and height > 0:
                if i < len(bbs_aug.bounding_boxes) - 1:
                    singleLabelAndBoxes = augBoxCordinate.label + " " + str(x_center) + " "  + str(y_center) + " " +  str(width)  + " " + str(height) + "\n"
                else:
                    singleLabelAndBoxes = augBoxCordinate.label + " " + str(x_center) + " "  + str(y_center) + " " +  str(width)  + " " + str(height)
                f.write(singleLabelAndBoxes)
            #print(singleLabelAndBoxes, i)


def cropPad(image, imagePath, bbs):
    seq = iaa.Sequential([iaa.CropAndPad(
                                        px=((0, -10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                                        pad_mode="linear_ramp",   
                                        pad_cval=(0, 150)
                                        )
                        ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "cropPad")
    
    return image_aug, bbs_aug

def prospectiveTransform(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.CropAndPad(percent=(-0.2, 0.2))])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "prosTrans")
    
    return image_aug, bbs_aug

def affinePadding(image, imagePath, bbs):
    seq = iaa.Sequential([ iaa.ShearX((0, 10)), iaa.Affine(translate_percent={"x": 0.1}, scale=random.uniform(0.4, 0.9)) ])
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "affinePadding")
    
    return image_aug, bbs_aug
  
 #   iaa.Affine(translate_percent={"x": 0.2}, scale=0.4),

def prospectiveCropPad(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.PerspectiveTransform(scale=(0.02, 0.1)),
                            iaa.CropAndPad(px=((0, 10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                            pad_mode="linear_ramp",   
                            pad_cval=(0, 150)
                            ) 
                            ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "prosCropPad")
    
    return image_aug, bbs_aug

def prospectiveTransformNoise(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.PerspectiveTransform(scale=(0.02, 0.1)),
                            iaa.CropAndPad(px=((0, 10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                            pad_mode="linear_ramp",   
                            pad_cval=(0, 150)
                            ),
                            iaa.AdditiveGaussianNoise(scale=(0, 0.15*255)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "prosCropPadNoise")
    
    return image_aug, bbs_aug

def prospectiveTransformBlur(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.PerspectiveTransform(scale=(0.02, 0.1)),
                            iaa.CropAndPad(px=((0, 10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                            pad_mode="linear_ramp",   
                            pad_cval=(0, 150)
                            ),
                            iaa.MotionBlur(k=random.randint(7, 15)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "prosCropPadBlur")
    
    return image_aug, bbs_aug

def gammaContrast(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((1.0, 3.0)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "gammaContrast")
    
    return image_aug, bbs_aug

def gammaContrastLow(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((0.5, 2.4)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "gammaContrast")
    
    return image_aug, bbs_aug
    
def sigmoidContrast(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.8)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "sigmoidContrast")
    
    return image_aug, bbs_aug
    

def gammaNoise(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((1.2, 2.0)), iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "gammaContrastNoise")
    
    return image_aug, bbs_aug

def gammaNoiseAffine(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((1.2, 2.0)), iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),  iaa.ShearX((0, 10)), iaa.Affine(translate_percent={"x": 0.1}, scale=random.uniform(0.4, 0.9)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "gammaNoiseAffine")
    
    return image_aug, bbs_aug
    
def prospectiveCropPadGammaContrast(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((1.2, 2.0)), 
                            iaa.PerspectiveTransform(scale=(0.02, 0.1)),
                            iaa.CropAndPad(px=((0, 10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                            pad_mode="linear_ramp",   
                            pad_cval=(0, 150)
                            ),  ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "prosCropGamma")
    
    return image_aug, bbs_aug
def extremeAug(image, imagePath, bbs):
    seq = iaa.Sequential([  iaa.GammaContrast((1.2, 2.0)), 
                            iaa.PerspectiveTransform(scale=(0.02, 0.1)),
                            iaa.CropAndPad(px=((0, 10), (0, 10), (0, int(image.shape[0]*.8)), (0, int(image.shape[1]*.8))),
                            pad_mode="linear_ramp",   
                            pad_cval=(0, 150)
                            ),
                            #iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                            iaa.MotionBlur(k=random.randint(2, 10)) ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "extreme")
    return image_aug, bbs_aug

def KeepSizeByResize(image, imagePath, bbs):
    seq = iaa.KeepSizeByResize(iaa.Resize({"height":640, "width":640}), keep_size=True)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "Resize")
    return image_aug, bbs_aug
    
def randomRotate(image, imagePath, bbs):
    seq = iaa.Affine(rotate=(-20, 20))
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    saveAug(image_aug, bbs_aug, imagePath, "randomRotate")
    return image_aug, bbs_aug

def yoloToimgaug(cordinates, dimension):
    x_center, y_center, yolo_width, yolo_height = cordinates
    image_width, image_height = dimension

    # Convert Yolo Format to Pascal VOC format
    box_width = yolo_width * image_width
    box_height = yolo_height * image_height
    # x_min = max(0, float(int(x_center * image_width - (box_width / 2))))
    # y_min = max(0, float(int(y_center * image_height - (box_height / 2))))
    # x_max = min(image_width - 1, float(int(x_center * image_width + (box_width / 2))))
    # y_max = min(image_height - 1, float(int(y_center * image_height + (box_height / 2))))
    x_min = float(int(x_center * image_width - (box_width / 2)))
    y_min = float(int(y_center * image_height - (box_height / 2)))
    x_max = float(int(x_center * image_width + (box_width / 2)))
    y_max = float(int(y_center * image_height + (box_height / 2)))

    #print(x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max,  y_max

def imgaug2Yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x_min = max(0, box[0])
    x_max = min(size[0] - 1, box[1])
    y_min = max(0, box[2])
    y_max = min(size[1] - 1, box[3])
    x = (x_min + x_max)/2.0
    y = (y_min + y_max)/2.0
    w = x_max - x_min
    h = y_max - y_min
    x_center = x*dw
    width = w*dw
    y_center = y*dh
    height = h*dh
    #print(x_center, width, y_center, height)
    
    return x_center,y_center,width,height

listofAug = [randomRotate, gammaContrast]
#listofAug = [prospectiveCropPad]
def readFiles(pathImg):
    
    pathBox = pathImg.replace(".jpg", ".txt")
    # print(pathImg, pathBox)
    image = imageio.imread(pathImg)
    image_width = image.shape[1]
    image_height = image.shape[0]
    with open(pathBox) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        #print(lines[0])
    bbox = []
    try:
        for line in lines:
            
            label, x_center, y_center, width, height = line.split(" ")
            #print(label, x_center, y_center, width, height )
            x_min, y_min, x_max, y_max = yoloToimgaug(( float(x_center), float(y_center), float(width), float(height)), (image_width, image_height) )
            #imgaug2Yolo([image_width, image_height], [x_min, y_min, x_max, y_max])
            
            bbox.append(BoundingBox(x_min, y_min, x_max, y_max, label))
        
        bbs = BoundingBoxesOnImage(bbox, shape=image.shape)
        #ia.imshow(bbs.draw_on_image(image, size=2))
    except:
        pass
    try:
        for l in range(len(listofAug)):
            image_aug, bbs_aug = listofAug[l](image, pathImg, bbs)
    except:
        pass
    return  

if __name__ == "__main__":
    pathImg = glob.glob('/home/fssv3/wjh/dataset/ThinkI/OD/Refined_mk5/Resize/train/*.jpg')
    # pathtxt = "/home/fssv3/wjh/dataset/ThinkI/OD/path_list/K390_refined_mk4+mk5_valid.txt"
    # pathImg = load_images(pathtxt)
    # path = '/home/fssv3/wjh/augmentation/test/20220630112009-129_frame_8.jpg'
    # num = 0
    # for i in tqdm(range(8)):
    #     readFiles(path)
    #     num += 1
    # print(len(pathImg))
    
    for path in tqdm(pathImg):
        readFiles(path)
