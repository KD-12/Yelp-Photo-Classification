#https://stackoverflow.com/questions/19285562/python-opencv-imread-displaying-image
from collections import defaultdict
import glob
import re
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

def check_img(img_name):
    print(img_name)
    try:
        path = glob.glob("C:\\Users\\komal\\pyex\\google_cloud\\input\\test_photos\\"+img_name)[0]
    except Exception as e:
        print('error: image not found')
        return
    img = cv2.imread(path)
    #new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.namedWindow(img_name)
    #cv2.resizeWindow(img_name, 300, 300)
    cv2.imshow(img_name, img)
    cv2.waitKey(8000)

def create_biz_to_photo_dict(path):
    biz_to_photo = defaultdict(list)
    df = pd.read_csv(path, dtype={0: str, 1:str})
    for (photo_id,biz_id) in df.values:
        biz_to_photo[biz_id].append(photo_id)
    return biz_to_photo

if __name__ == "__main__":
    print("Testing Yelp Classification")
    test_biz_to_photo = create_biz_to_photo_dict("./test_photo_to_biz.csv")
    photo_to_label = {}
    with open("./result_photo_label", "w") as f1:
        with open("./result_lr_nn","r") as f:
            for e, line in enumerate(f):
                #if e >6:
                #    break
                #print(line)
                line_list = line.rstrip().split(",")
                f1.write(" ".join(test_biz_to_photo[line_list[0]])+" "+" ".join(line_list))
                for photo_id in test_biz_to_photo[line_list[0]]:
                #    print(photo_id)
                    photo_to_label[photo_id] = line
    print("Trying to predict labels:")
    print("0: good_for_lunch")
    print("1: good_for_dinner")
    print("2: takes_reservations")
    print("3: outdoor_seating")
    print("4: restaurant_is_expensive")
    print("5: has_alcohol")
    print("6: has_table_service")
    print("7: ambience_is_classy")
    print("8: good_for_kids")
    image_id = input("Enter an image_id (1, 471992) which you want to check:")
    check_img(str(image_id) + '.jpg')
    print(photo_to_label[image_id])
