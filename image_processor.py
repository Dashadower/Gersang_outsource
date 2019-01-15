# -*- coding:utf-8 -*-
import cv2, numpy as np, time, random, math, pytesseract

class Preprocessor:
    def __init__(self, background_img="images/background.png"):
        self.roi_area = [48, 185, 980, 656]  # x1, y1, x2, y2
        self.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        self.bg_img = cv2.imread(background_img, cv2.IMREAD_COLOR)[self.roi_area[1]:self.roi_area[3], self.roi_area[0]:self.roi_area[2]]
        self.bg_hsv = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2HSV)

        self.cluster_line_maxdist = 35

        self.cluster_hough_lines = []
        self.cluster_visited_lines = []
        self.cluster_linelist = []
        self.cluster_line_groups = []
        self.cluster_in_class = []

        self.cluster_contour_maxdist = 50
        self.cluster_contours = []
        self.cluster_contour_groups = []
        self.cluster_contour_centroids_in_class = []

        self.output_width = 40
        self.output_height = 60

    def crop_roi(self, input_img):
        return input_img[self.roi_area[1]:self.roi_area[3], self.roi_area[0]:self.roi_area[2]]

    def remove_background(self, input_img):
        """
        Removes background using self.bg_img
        :param input_img: BGR image
        :return: BGR image with background portions set to 0
        """
        reconstruction_image = np.zeros([input_img.shape[0], input_img.shape[1], 3], dtype=np.uint8)
        hue_offset = 5

        input_img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

        for y in range(input_img_hsv.shape[0]):
            for x in range(input_img_hsv.shape[1]):
                sp = input_img_hsv[y][x]
                bp = self.bg_hsv[y][x]

                if bp[0] - hue_offset <= sp[0] <= bp[0] + hue_offset:
                    if sp[1] >= 109:
                        reconstruction_image[y][x] = input_img[y][x]
                elif sp[1] <= 90:
                    if sp[0] >= 67:
                        reconstruction_image[y][x] = input_img[y][x]
                    elif sp[2] >= 125 and sp[1] >= 20:
                        reconstruction_image[y][x] = input_img[y][x]
                else:
                    reconstruction_image[y][x] = input_img[y][x]

        return reconstruction_image

    def threshold_and_preprocess(self, input_image):
        """
        thresholds input image and performs preprocesing operations
        :param input_image: BGR image
        :return: Binary image with text as white
        """

        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(input_gray, (3, 3), 0)
        sharpened = cv2.addWeighted(blur, 1.5, input_gray, -0.5, 0)
        denoised = cv2.fastNlMeansDenoising(sharpened, h=11, templateWindowSize=7, searchWindowSize=21)

        #ret, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(denoised, 27, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        #operated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #operated = cv2.morphologyEx(operated, cv2.MORPH_CLOSE, kernel)
        operated = cv2.erode(thresh, kernel)
        operated = cv2.dilate(operated, kernel)
        operated = operated
        contour_area_threshold = 20
        _, contours, _ = cv2.findContours(operated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bgred = cv2.cvtColor(operated, cv2.COLOR_GRAY2BGR)
        for ct in contours:
            if cv2.contourArea(ct) <= contour_area_threshold:
                cv2.drawContours(bgred, [ct], -1, (0,0,0), -1)
        #operated = cv2.cvtColor(bgred, cv2.COLOR_BGR2GRAY)
        #operated = cv2.erode(operated, kernel)
        return operated


    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def cluster_lines(self, input_img):
        """
        Clusters white lines in binary images by distance between each other, using self.cluster_line_maxdist
        :param input_img: BINARY input image with linese
        :return: a list of sublists x, where an element of x is a tuple ((x1,y1,x2,y1), (cx,cy)), point c is center of line
        """
        self.cluster_hough_lines = []
        self.cluster_visited_lines = []
        self.cluster_linelist = []
        self.cluster_line_groups = []
        self.cluster_in_class = []

        self.cluster_hough_lines = cv2.HoughLinesP(input_img, rho=7, theta= np.pi / 180, threshold=10, minLineLength=8, maxLineGap=2)

        for lv in self.cluster_hough_lines:
            self.cluster_linelist.append((lv[0], (int((lv[0][0] + lv[0][2]) / 2), int((lv[0][1] + lv[0][3]) / 2))))

        for line in self.cluster_linelist:
            coords = line[0]
            centroid = line[1]
            if not self.cluster_line_groups or centroid not in self.cluster_in_class:
                group_obj = []
                resolved = False
                current_line = line
                self.cluster_traverse(current_line, group_obj)
                self.cluster_line_groups.append(group_obj)

        return self.cluster_line_groups

    def cluster_check_class(self, line_obj):
        for gp in self.cluster_line_groups:
            for lin in gp:
                if (lin[0] == line_obj[0]).all():
                    return True

        return False

    def cluster_find_neighbor(self,line_obj):
        return_list = []
        for lt in self.cluster_linelist:
            if self.distance(line_obj[1], lt[1]) <= self.cluster_line_maxdist and lt[1] not in self.cluster_in_class and not (
                    line_obj[0] == lt[0]).all():
                return_list.append(lt)
        return return_list

    def cluster_traverse(self, line_obj, group):
        group.append(line_obj)
        #self.cluster_visited_lines.append(line_obj[1])
        self.cluster_in_class.append(line_obj[1])
        nbhh = self.cluster_find_neighbor(line_obj)
        if not nbhh:
            return 0
        else:
            for n2 in nbhh:
                self.cluster_traverse(n2, group)

    def find_text_contour_hierachy(self, input_img):
        """
        An alternative attempt to isolate text, instead of using line clustering. Current method is to find contours
        and resolve points using approxPolyDP
        :param input_img: binary image, white as features
        :return: list of tuple (lx, h) where lx is a list of cluster_contour object within group,
        and h a cluster_contour object with the highest y value
        """
        _, contours, _ = cv2.findContours(input_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[::-1]
        self.cluster_contours = []
        self.cluster_contour_groups = []
        self.cluster_contour_centroids_in_class = []
        # ret_list is a list of tuple v where
        # v = (points, (c_x, c_y), (l_x, l_y), ct) where points is a list of points, point c which is mean point, point
        # l which is located lowest on the screen, meaning highest y value. and ct the contour object
        for contour in contours:
            """if cv2.contourArea(contour) <= 10:
                continue"""
            points = cv2.approxPolyDP(contour, 0.3 * cv2.arcLength(contour, True), True)
            contour_points = []
            center_x = 0
            center_y = 0
            lowest_x = 0
            lowest_y = 0
            for j in points:

                p_x, p_y = j[0]
                if p_y > lowest_y:
                    lowest_x = p_x
                    lowest_y = p_y
                center_x += p_x
                center_y += p_y
                contour_points.append((p_x, p_y))

            self.cluster_contours.append((contour_points, (int(center_x/len(points)), int(center_y/len(points))), (lowest_x, lowest_y), contour))

        for contour in self.cluster_contours:
            coords = contour[0]
            centroid = contour[1]
            lowest = contour[2]
            if not self.cluster_contour_groups or centroid not in self.cluster_contour_centroids_in_class:
                group_obj = [[], contour]
                resolved = False
                self.cluster_contour_traverse(contour, group_obj)
                points = []
                for cv in group_obj[0]:
                    points.extend(cv[0])
                points = np.array(points)
                if len(group_obj[0]) >= 2 and cv2.contourArea(points) >= 50:
                    self.cluster_contour_groups.append(group_obj)

        return self.cluster_contour_groups

    def cluster_contour_traverse(self, current, cluster_group):
        cluster_group[0].append(current)

        if current[2][1] > cluster_group[1][2][1]:
            cluster_group[1] = current

        self.cluster_contour_centroids_in_class.append(current[1])
        neighbors = self.cluster_contour_find_neighbor(current)

        if not neighbors:
            return 0
        else:
            for neighbor in neighbors:
                self.cluster_contour_traverse(neighbor, cluster_group)

    def cluster_contour_find_neighbor(self, cont):
        return_list = []
        for lt in self.cluster_contours:
            if self.distance(cont[1], lt[1]) <= self.cluster_contour_maxdist and lt[1] not in self.cluster_contour_centroids_in_class and not (
                    cont[2] == lt[2]):
                return_list.append(lt)
        return return_list

    def find_baseline_and_deskew_from_contour(self, input_img, contour_group):
        """
        Given binary input_image and contour cluster data contour_group, return list of tuples(img, point) where image
        is binary deskewed image with size of self.output_width and height, and point which is center point of reference
        line.
        :param input_img: Binary image where background is black, features are white
        :param line_group: same format as output of self.find_text_contour_hierachy
        :return: list of tuples (img, point) where image is binary deskewed image where features are black,
        with width, height of self.output_width and self.output_height, point which is center point coordinate,
        relative to ROI region, of reference line.
        """
        return_list = []
        for group in contour_group:
            representation_contour = group[1]
            sorted_points = sorted(representation_contour[0], key=lambda x: x[0])
            r_centroid = (int(representation_contour[1][0]), int(representation_contour[1][1]))
            r_1 = sorted_points[0]

            r_2 = sorted_points[1]

            #if r_1[1] > r_2[1]:

            angle = math.atan2(r_1[1] - r_2[1], r_1[0] - r_2[0]) * 180.0 / math.pi
            """if angle < 0.0:
                angle += 360.0
            if angle > 180:
                angle = 360 - angle"""
            angle1 = (angle + 360) % 360
            angle2 = (angle1 + 180) % 360
            print(angle1, angle2)
            #kernel = np.array([[-1,-1,-1], [-1,1,-1], [-1,-1,-1]], dtype=np.uint8)
            #input_img = cv2.filter2D(input_img, -1 ,kernel)

            t_img = cv2.bitwise_not(input_img).copy() # now features/contours are black on white background
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(t_img, [representation_contour[3]], -1, (255, 255, 255), -1)
            t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)

            rows, cols = t_img.shape
            root_mat = cv2.getRotationMatrix2D(r_centroid, angle, 1)
            rotated = cv2.warpAffine(t_img, root_mat, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

            cropped_result = rotated[
                             max(int(r_centroid[1] - self.output_height), 0):min(int(r_centroid[1] + 8), rotated.shape[0]),
                             max(0, int(r_centroid[0] - self.output_width)):min(rotated.shape[1],
                                                                      int(r_centroid[0] + self.output_width))]

            root_mat_2 = cv2.getRotationMatrix2D(r_centroid, angle2, 1)
            rotated_2 = cv2.warpAffine(t_img, root_mat_2, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

            cropped_result_2 = rotated_2[
                             max(int(r_centroid[1] - self.output_height), 0):min(int(r_centroid[1] + 8),
                                                                                 rotated.shape[0]),
                             max(0, int(r_centroid[0] - self.output_width)):min(rotated.shape[1],
                                                                                int(r_centroid[0] + self.output_width))]

            if cv2.countNonZero(cropped_result) > cv2.countNonZero(cropped_result_2):
                cropped_result = cropped_result_2
            """cropped_result = cv2.resize(cropped_result,None, fx=2.0, fy=2.0)

            kernel = np.ones((5, 5), dtype=np.uint8)
            cropped_result = cv2.erode(cropped_result, kernel)"""
            """cropped_result = rotated[
                             int(r_centroid[1] - self.output_height):int(r_centroid[1] + 4),
                             int(r_centroid[0] - self.output_width):int(r_centroid[0] + self.output_width)]"""

            #cropped_result = cv2.resize(cropped_result, (0, 0), fx=2, fy=2)


            return_list.append((cropped_result, r_centroid))

        return return_list

    def find_baseline_and_deskew(self, input_img, line_group):
        """
        Given binary input_image and cluster data line_group, return list of tuples(img, point) where image is binary
        deskewed image with size of self.output_width and height, and point which is center point of reference line.
        :param input_img: Binary image where background is black, features are white
        :param line_group: same format as output of self.cluster_lines
        :return: list of tuples (img, point) where image is binary deskewed image where features are black,
        with width, height of self.output_width and self.output_height, point which is center point coordinate,
        relative to ROI region, of reference line.
        """
        return_list = []
        for group in self.cluster_line_groups:

            sorted_group = sorted(group, key=lambda x: max(x[0][1], x[0][3]))
            representation_line = sorted_group[0]
            r_c = representation_line[0]
            r_centroid = representation_line[1]
            angle = math.atan2(r_c[3] - r_c[1], r_c[2] - r_c[0]) * 180.0 / math.pi
            #kernel = np.array([[-1,-1,-1], [-1,1,-1], [-1,-1,-1]], dtype=np.uint8)
            #input_img = cv2.filter2D(input_img, -1 ,kernel)

            t_img = cv2.bitwise_not(input_img).copy()
            rows, cols = t_img.shape
            root_mat = cv2.getRotationMatrix2D(representation_line[1], angle, 1)
            rotated = cv2.warpAffine(t_img, root_mat, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))


            cropped_result = rotated[
                             max(int(r_centroid[1] - self.output_height), 0):min(int(r_centroid[1] + 8), rotated.shape[0]),
                             max(0, int(r_centroid[0] - self.output_width)):min(rotated.shape[1],
                                                                      int(r_centroid[0] + self.output_width))]
            """cropped_result = rotated[
                             int(r_centroid[1] - self.output_height):int(r_centroid[1] + 4),
                             int(r_centroid[0] - self.output_width):int(r_centroid[0] + self.output_width)]"""

            #cropped_result = cv2.resize(cropped_result, (0, 0), fx=2, fy=2)


            return_list.append((cropped_result, r_centroid))

        return return_list

    def run_tesseract(self, input_img, lang="eng", config="--psm 10"):
        """
        Runs tesseract and returns result
        :param input_img: image to feed tesseract
        :return: tesseract output string
        """
        return pytesseract.image_to_string(input_img,lang=lang, config=config)

    def solve(self, input_img):
        """
        Master function to calculate coordinate and value of characters in image. This function will only return
        values and coordinates; input and sorting is not handled.
        :param input_img: BGR image of GerSang window in 1024x768
        :return: list of tuple (coord, value) where coord tuple (x,y) is a clickable region within input_image,
        value is computed value of character within coord region.
        """
        input_img_background_removed = self.remove_background(input_img)
        processed_img = self.threshold_and_preprocess(input_img_background_removed)
        cluster_data = self.cluster_lines(processed_img)

        deskewed_and_cropped = self.find_baseline_and_deskew(processed_img, cluster_data)

        return_arr = []
        for data in deskewed_and_cropped:
            image, coord = data
            tesseract_result = self.run_tesseract(image)
            if tesseract_result:
                output = tesseract_result
            else:
                output = ""
            return_arr.append((coord, output))

        return return_arr

if __name__ == "__main__":
    prc = Preprocessor()
    for x in range(2, 12):
        img = prc.crop_roi(cv2.imread("images/source/%d.png"%(x), cv2.IMREAD_COLOR))
        representation = img.copy()
        tx = prc.remove_background(img)
        processed = prc.threshold_and_preprocess(tx)
        cv2.imshow("processed", processed)

        contour_groups = prc.find_text_contour_hierachy(processed)

        for group in contour_groups:
            color = (random.randrange(0, 254), random.randrange(0, 254), random.randrange(0, 254))
            #print(group[0])
            #print(group[1])
            contours, lowest_contour = group
            cv2.circle(tx, lowest_contour[1], 3, (0,0,255), -1)

            cv2.circle(tx, lowest_contour[0][0], 2, (0,0,255), -1)
            cv2.circle(tx, lowest_contour[0][1], 2, (0,0,255), -1)

        cv2.imshow("bg_removal", tx)
        cv2.imshow("processed", processed)
        cv2.waitKey(0)

        """for deskewed, ct in prc.find_baseline_and_deskew_from_contour(processed, contour_groups):
            result = prc.run_tesseract(deskewed)
            print(ct)
            cv2.putText(representation, result, ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("dt", representation)
        cv2.waitKey(0)"""





