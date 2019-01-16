# -*- coding:utf-8 -*-
import cv2, numpy as np, glob, time

bg_img_path = "images/background.png"
test_img_dir = "images/source"

bg_img = cv2.imread(bg_img_path, cv2.IMREAD_COLOR)

for img in glob.glob(test_img_dir+"/*.png"):
    img_obj = cv2.imread(img, cv2.IMREAD_COLOR)
    subtracted = cv2.subtract(img_obj, bg_img)
    gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    cv2.imshow("", gray)
    cv2.waitKey(0)