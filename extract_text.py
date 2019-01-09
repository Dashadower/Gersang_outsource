# -*- coding:utf-8 -*-
import cv2, numpy as np, time, random, math, pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

img_roi = [48, 185, 980, 656]  # x1, y1, x2, y2
src_img_dir = "images/source/7.png"
bg_img = cv2.imread("images/background.png", cv2.IMREAD_COLOR)[img_roi[1]:img_roi[3], img_roi[0]:img_roi[2]]
# The background of the area is constant. So I have used a reference background image and removed pixels which have a similar H value as the background

bg_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
src_img = cv2.imread(src_img_dir, cv2.IMREAD_COLOR)[img_roi[1]:img_roi[3], img_roi[0]:img_roi[2]]
# This image is the image where letters are placed on top of the background image

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

print("duration", time.time()-start_time)

mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3,3))
opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, (3,3))
opened = cv2.erode(opened, (3,3))
opened = cv2.dilate(opened, (3,3))
opened = cv2.dilate(opened, (5, 5))
opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, (3,3))
opened = cv2.erode(opened, (3,3))
opened = cv2.erode(opened, (5,5))
opened = cv2.dilate(opened, (3,3))
opened = cv2.dilate(opened, (3,3))
opened = cv2.dilate(opened, (3,3))
final_img = opened
#edges = cv2.Canny(final_img, 0, 255)

max_line_cluster_distance = 30
img_width = 40
img_height = 60
def distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


lines = cv2.HoughLinesP(final_img, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=2)
linelist = []
for lv in lines:
    linelist.append((lv[0], ((lv[0][0]+lv[0][2])/2, (lv[0][1]+lv[0][3])/2)))
line_groups = []
visited = []
def find_neighbor(lin_obj):
    return_list = []
    for lt in linelist:
        if distance(lin_obj[1], lt[1]) <= max_line_cluster_distance and not check_class(lt) and not (lin_obj[0]==lt[0]).all():
            if lt[1] not in visited:
                return_list.append(lt)
    return return_list

def check_class(lin_obj):
    for gp in line_groups:
        for lin in gp:
            if (lin[0]==lin_obj[0]).all():
                return True

    return False

def spread(lin_obj, group):
    group.append(lin_obj)
    visited.append(lin_obj[1])
    nbhh = find_neighbor(lin_obj)
    if not nbhh:
        return 0
    else:
        for n2 in nbhh:
            spread(n2, group)

for line in linelist:
    coords = line[0]
    centroid = line[1]
    if not line_groups or not check_class(line):
        print("check class:", line, "is false")
        group_obj = []
        resolved = False
        current_line = line
        spread(current_line, group_obj)
        line_groups.append(group_obj)

print("Groups", len(line_groups))
for group in line_groups:
    color = (random.randrange(0, 254),random.randrange(0, 254),random.randrange(0, 254))
    sorted_group = sorted(group, key = lambda x: max(x[0][1], x[0][3]), reverse=True)
    cv2.line(src_img, (sorted_group[0][0][0], sorted_group[0][0][1]), (sorted_group[0][0][2], sorted_group[0][0][3]), (0, 255, 255), 5)
    representation_line = sorted_group[0]
    r_c = representation_line[0]
    r_centriod = representation_line[1]
    print("r centroid", r_centriod)
    angle = math.atan2(r_c[3]-r_c[1], r_c[2]-r_c[0]) * 180.0 / math.pi
    """connectivity = 8
    min_size = 5
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_img, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]
    num_labels = num_labels - 1
    for i in range(0, num_labels):
        if sizes[i] >= min_size:
            final_img[labels == i + 1] = 0"""
    t_img = cv2.bitwise_not(final_img).copy()
    rows, cols = t_img.shape
    root_mat = cv2.getRotationMatrix2D(representation_line[1], angle, 1)
    result = cv2.warpAffine(t_img, root_mat,(cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    cropped_result = result[max(int(r_centriod[1]-img_height), 0):min(int(r_centriod[1]+4), result.shape[0]), max(0,int(r_centriod[0]-img_width)):min(result.shape[1],int(r_centriod[0]+img_width))]
    cropped_result = cv2.resize(cropped_result, (0,0), fx = 2, fy=2)

    #cropped_result = cv2.medianBlur(cropped_result, 7)

    """total_pixels = cropped_result.shape[0] * cropped_result.shape[1]
    black_pixels = total_pixels - cv2.countNonZero(cropped_result)
    if black_pixels/total_pixels * 100 >= 10:
        res = pytesseract.image_to_string(cropped_result, lang="eng", config="--psm 10")
        print("tesseract:", res)
    else:
        res = pytesseract.image_to_string(cropped_result, lang="eng", config="--psm 10")
        print("tesseract:", res)
        print("skip", (total_pixels-black_pixels)/total_pixels * 100)"""
    res = pytesseract.image_to_string(cropped_result, lang="eng", config="--psm 10")
    print("tesseract:", res)
    cv2.imshow("", cropped_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    linecnt = 0
    for l in sorted_group:
        cd = l[0]
        if linecnt != 0:
            cv2.line(src_img, (cd[0], cd[1]), (cd[2], cd[3]), color, 2)
        linecnt += 1


#cv2.imshow("can", edges)




#cv2.drawContours(src_img, fixed_contours, -1, (0,255,0), 2)
cv2.imshow("src", src_img)
cv2.imshow("", cv2.bitwise_not(final_img))

cv2.waitKey(0)
cv2.destroyAllWindows()