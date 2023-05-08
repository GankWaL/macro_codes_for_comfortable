import cv2

cls_color = {0:(0,255,0), 1:(255,0,0), 2:(0, 255, 255), 3:(255, 0, 255), 4:(255, 255, 255), 5:(255, 255, 0)}
cls_name = {0:"open_eye", 1:"close_eye", 2:"cigarette", 3:"phone", 4:"face", 5:"mask"}

def draw_yolo_box(path):
    # Load the image
    image = cv2.imread(path)
    txt_path = path.replace(".jpg", ".txt")
    
    # image = image[40:680, 640:1280]
    # image = cv2.resize(image, (320,320))
    # height, width, c = image.shape
    height = 640
    width = 640
    show_image = image.copy()
    
    bbox = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    
        for line in lines:            
            cls, cx, cy, w, h = line.split(" ")            
            cls = int(cls)
            box_width = float(w) * width
            box_height = float(h) * height

            x_min = int(float(cx) * width - (box_width / 2)) + 640
            y_min = int(float(cy) * height - (box_height / 2)) + 40
            x_max = int(float(cx) * width + (box_width / 2)) + 640
            y_max = int(float(cy) * height + (box_height / 2)) + 40
            
            bbox.append((cls, x_min, y_min, x_max, y_max))
            show_image = cv2.rectangle(show_image, (x_min, y_min), (x_max, y_max), cls_color[cls], 1)
            show_image = cv2.putText(show_image, cls_name[cls], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color[cls], 1)
            
            if cls == 4:
                hp_x_min = int(float(cx) * width - (box_width * 0.7)) + 640
                hp_y_min = int(float(cy) * height - (box_height * 0.7)) +40
                hp_x_max = int(float(cx) * width + (box_width * 0.7)) + 640
                hp_y_max = int(float(cy) * height + (box_height * 0.7)) + 40
                
                ciga_x_min = int(float(cx) * width - (box_width * 0.75))
                ciga_y_min = int(float(cy) * height - (box_height * 0.25))
                ciga_x_max = int(float(cx) * width + (box_width * 0.75))
                ciga_y_max = int(float(cy) * height + (box_height * 0.53))
                
                ph_x_min = int(float(cx) * width - (box_width * 1.5))
                ph_y_min = int(float(cy) * height - (box_height * 0.75))
                ph_x_max = int(float(cx) * width + (box_width * 1.5))
                ph_y_max = int(float(cy) * height + (box_height * 0.75))
                
                show_image = cv2.rectangle(show_image, (hp_x_min, hp_y_min), (hp_x_max, hp_y_max), cls_color[4], 2)
                show_image = cv2.putText(show_image, "Headpose " + cls_name[4], (hp_x_min, hp_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color[4], 2)
                
                show_image = cv2.rectangle(show_image, (709, 215), (989, 583), (0, 0 ,255), 1)
                show_image = cv2.putText(show_image, "before refine rect face", (709, 215 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                show_image = cv2.rectangle(show_image, (670, 216), (1038, 584), (255, 0 ,0), 1)
                show_image = cv2.putText(show_image, "after refine rect face", (670, 216 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # show_image = cv2.rectangle(show_image, (ciga_x_min, ciga_y_min), (ciga_x_max, ciga_y_max), cls_color[2], 2)
                # show_image = cv2.putText(show_image, "ROI of " + cls_name[2], (ciga_x_min, ciga_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color[2], 2)
                
                # show_image = cv2.rectangle(show_image, (ph_x_min, ph_y_min), (ph_x_max, ph_y_max), cls_color[3], 2)
                # show_image = cv2.putText(show_image, "ROI of " + cls_name[3], (ph_x_min, ph_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls_color[3], 2)
    
    show_image = cv2.imshow('draw yolo box to image', show_image)
    cv2.waitKey(0)

def draw_face_box(path):
    image = cv2.imread(path)
    image = cv2.rectangle(image, (709, 215), (989, 583), (0, 0 ,255), 1)
    image = cv2.putText(image, "before refine rect face", (709, 215 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    image = cv2.rectangle(image, (670, 216), (1038, 584), (255, 0 ,0), 1)
    image = cv2.putText(image, "after refine rect face", (670, 216 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    image = cv2.imshow('draw facebox for headpose', image)
    cv2.waitKey(0)
    
    
if __name__ == "__main__":
    path = "D:/test/DSM/think-i/YUV_bbox_v53/face/FS_DSM_yuv_source_image_330.jpg"
    draw_yolo_box(path)
    # draw_face_box(path)
    