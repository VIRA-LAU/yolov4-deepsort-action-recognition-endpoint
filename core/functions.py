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
    if classname =='person':
        if xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
            cropped_img = img[int(ymin) - 35:int(ymax) + 35, int(xmin) - 35:int(xmax) + 35]
            img_name = classname + '_' + str(id) + '_' + str(frame_img_number) + '.png'
            img_path = os.path.join(path, img_name)
            cv2.imwrite(img_path, cropped_img)
            frame_img_number = frame_img_number+1

    return frame_img_number
    #create dictionary to hold count of objects for image name
    # counts = dict()
    # for i in range(num_objects):
    #     # get count of class for part of image name
    #     class_index = int(classes[i])
    #     class_name = class_names[class_index]
    #     if class_name in allowed_classes:
    #         counts[class_name] = counts.get(class_name, 0) + 1
    #         # get box coords
    #         xmin, ymin, xmax, ymax = boxes[i]
    #         # crop detection from image (take an additional 5 pixels around all edges)
    #         cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
    #         # construct image name and join it to path for saving crop properly
    #         img_name = class_name + '_' + str(counts[class_name]) + '.png'
    #         img_path = os.path.join(path, img_name)
    #         # save image
    #         cv2.imwrite(img_path, cropped_img)
    #     else:
    #         continue
