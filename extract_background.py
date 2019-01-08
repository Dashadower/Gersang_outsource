# -*- coding:utf-8 -*-
import glob, os
import cv2, numpy as np
from collections import Counter
src_img_dir = "images/source"
os.chdir(src_img_dir)
img_size = [1030, 793]
color_stats = []
for g in range(img_size[1]):
    color_stats.append([])
    for t in range(img_size[0]):
        color_stats[g].append([])
#print(color_stats[0][0])
for img_path in glob.glob("*.png"):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print("processing", img_path)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if x == 100 and y == 100:
                print(img[y][x].tolist())
            color_stats[y][x].append(tuple(img[y][x].tolist()))

#print(color_stats[100][100])
result_img = np.zeros([img_size[1], img_size[0], 3], dtype=np.uint8)
for y in range(img_size[1]):
    for x in range(img_size[0]):
        dta = Counter(color_stats[y][x])
        dv = np.array(dta.most_common(1)[0][0], dtype=np.uint8)
        for channel in range(3):
            result_img[y][x][channel] = dv[channel]

cv2.imshow("", result_img)
cv2.waitKey(0)
cv2.imwrite("background.png", result_img)
cv2.destroyAllWindows()




