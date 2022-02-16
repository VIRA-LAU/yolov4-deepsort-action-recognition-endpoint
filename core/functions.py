import cv2
from core.utils import read_class_names
from core.config import cfg
import os


# function for cropping each detection and saving as new image
def crop_objects(img, bbox, id, classname, path, allowed_classes, frame_img_number):
    print(bbox, id, classname)

    xmin, ymin, xmax, ymax = bbox
    print(xmin,ymin,xmax,ymax)



    print(frame_img_number)
    if classname !='basketball':
        if xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
            cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
            img_name = classname + '_' + str(id) + '_' + str(frame_img_number) + '.png'
            img_path = os.path.join(path, img_name)
            cv2.imwrite(img_path, cropped_img)
            frame_img_number = frame_img_number+1

    return frame_img_number

