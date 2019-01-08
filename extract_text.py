# -*- coding:utf-8 -*-
import cv2, numpy as np, time
img_roi = [48, 191, 980, 656]  # x1, y1, x2, y2
src_img_dir = "images/source/10.png"
bg_img = cv2.imread("images/background.png", cv2.IMREAD_COLOR)[img_roi[1]:img_roi[3], img_roi[0]:img_roi[2]]
bg_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
src_img = cv2.imread(src_img_dir, cv2.IMREAD_COLOR)[img_roi[1]:img_roi[3], img_roi[0]:img_roi[2]]
src_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
mask = np.zeros([src_img.shape[0], src_img.shape[1], 3], dtype=np.uint8)

offset = 3
start_time = time.time()
for y in range(src_img.shape[0]):
    for x in range(src_img.shape[1]):
        sp = src_hsv[y][x]
        bp = bg_hsv[y][x]

        if bp[0]-offset <= sp[0] <= bp[0]+offset:
            if sp[1] >= 109:
                mask[y][x] = src_img[y][x]
        elif sp[1] <= 90:
            if sp[0] >= 67:
                mask[y][x] = src_img[y][x]
            elif sp[2] >= 125 and sp[1] >= 20:
                mask[y][x] = src_img[y][x]
        else:
            mask[y][x] = src_img[y][x]
        """if sp[1] >= 60 and sp[2] >= 60:
            mask[y][x] = src_img[y][x]
            #mask[y][x] = conv"""

print("duration", time.time()-start_time)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
#mask[:,:,2] = 255
mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3))
opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, (3,3))
opened = cv2.erode(opened, (3,3))
opened = cv2.dilate(opened, (3,3))
opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, (3,3))
final_img = opened
lines = cv2.HoughLinesP(final_img, 1, np.pi/180, 24, minLineLength=15, maxLineGap=2)
line_list = []
"""for line in lines:
        coords = line[0]
        dvy = src_img.copy()
        line_length = np.sqrt((coords[0]-coords[2])**2 + (coords[1]-coords[3])**2)
        angle = np.rad2deg(np.arctan2(coords[3]-coords[1], coords[2]-coords[0]))

        line_list.append((coords, line_length, angle))
        print(coords, line_length, angle)
        cv2.line(dvy, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
        cv2.imshow("bg", dvy)
        cv2.waitKey(0)"""
fixed_contours = []
_, contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:

    if cv2.contourArea(cnt) >= 3:
        fixed_contours.append(cnt)

cv2.drawContours(src_img, fixed_contours, -1, (0,255,0), 2)
cv2.imshow("src", src_img)
cv2.imshow("", final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()