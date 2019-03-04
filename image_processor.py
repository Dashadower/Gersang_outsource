# -*- coding:utf-8 -*-
import cv2, numpy as np, time, random, math, pytesseract, imutils

class Preprocessor:
    def __init__(self, background_img="images/background.png"):
        self.roi_area = [48, 185, 980, 656]  # x1, y1, x2, y2
        self.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        self.bg_img = cv2.imread(background_img, cv2.IMREAD_COLOR)[self.roi_area[1]:self.roi_area[3], self.roi_area[0]:self.roi_area[2]]
        self.bg_hsv = cv2.cvtColor(self.bg_img, cv2.COLOR_BGR2HSV)


        self.cluster_contour_maxdist = 50
        self.cluster_contours = []
        self.cluster_contour_groups = []
        self.cluster_contour_centroids_in_class = []

        self.output_width = 30  # actual output width is output_width * 2
        self.output_height = 60

    def crop_roi(self, input_img):
        return input_img[self.roi_area[1]:self.roi_area[3], self.roi_area[0]:self.roi_area[2]]

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def remove_background(self, input_img):
        """
        Removes background using self.bg_img
        :param input_img: BGR image
        :return: BGR image with background portions set to 0
        """

        hue_offset = 7
        saturation_offet = 115
        """
        An explanation on how background is removed.
        Currently the remover uses two conditions to remove background image, since noise is present on the background
        and using just cv2.sobtract() returns very poor results.
        Condition 1: If input_img[y][x] is within hue_offset range of bg_img[y][x], that pixel is considered positive
        for condition 1
        Condition 2: if saturation value of input_img[y][x] is less or equal than saturation_offset, that pixel is
        considered positive for condition 2
        
        If condition 1 and 2 are both positive for give pixel (x,y), (x,y) is replaced to (0,0,0), 
        else (x,y) = input_img[y][x]
        """

        input_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        inp_h, inp_s, inp_v = cv2.split(input_hsv)
        bg_h = cv2.split(self.bg_hsv)[0]
        mask1 = np.logical_and((inp_h>= bg_h-hue_offset), (inp_h<=bg_h+hue_offset))
        mask2 = inp_s <= saturation_offet

        delta = np.where(np.reshape(np.logical_and(mask1, mask2), inp_h.shape+(1,)), np.array([0, 0, 0], dtype=np.uint8), input_img)
        return delta

    def preprocess_alt(self, input_image):
        """
        This function handle preprocessing of image. If a background removed BGR image imput_image is given,
        this function will remove noise and threshold the image to have features/letters as white and others black.
        Note this operation is not perfect and can remove parts of features or keep noise
        :param input_image: BGR image with background removed
        :return: Binary grayscale image with features as white
        """
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        #_, threshed = cv2.threshold(input_gray, 50, 255, cv2.THRESH_BINARY)
        _, threshed = cv2.threshold(input_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # use otsu??

        denoised = cv2.fastNlMeansDenoising(threshed, h=17, templateWindowSize=7, searchWindowSize=21)

        return denoised

    def find_text_contour_hierachy(self, input_img):
        """
        This function groups nearby contours into groups and finds the contour which is deemed to be the baseline
        How it works is explained more in detail below
        :param input_img: binary image, white as features
        :return: list of tuple (lx, h) where lx is a list of cluster_contour object within group,
        and h a cluster_contour object with the highest y value

        cluster_contour object: tuple (l_p, center, lowest, ct)
        l_p : list of points(x,y) within contour
        center: tuple (x,y) of calculated mean point of contour
        lowest: tuple(x,y) of point which has highest y value = lowest on screen
        ct: contour object of countour returned by findContours
        """

        # This part is for support for newer OpenCV versions
        try:
            contours, _ = cv2.findContours(input_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            _, contours, _ = cv2.findContours(input_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[::-1]

        # Reinitialize temporary class variables
        self.cluster_contours = []
        self.cluster_contour_groups = []
        self.cluster_contour_centroids_in_class = []

        # Iterate over all contours and precalculate mean and lowest point
        for contour in contours:
            if cv2.contourArea(contour) <= 10:
                continue
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

        # Iterate over cluster_contours and group contours recursively using set self.cluster_contour_maxdist as threshold
        for contour in self.cluster_contours:
            centroid = contour[1]
            if not self.cluster_contour_groups or centroid not in self.cluster_contour_centroids_in_class:
                group_obj = [[], contour]
                # group_obj is cluster_group representing a single contour group passed to cluster_contour_traverse

                self.cluster_contour_traverse(contour, group_obj)

                points = []
                for cv in group_obj[0]:
                    points.extend(cv[0])
                points = np.array(points)

                if len(group_obj[0]) >= 2 and cv2.contourArea(points) >= 50:
                    self.cluster_contour_groups.append(group_obj)

        return self.cluster_contour_groups

    def cluster_contour_traverse(self, current, cluster_group):
        """
        Recursively group nearby contours.
        :param current: current cluster_contour object
        :param cluster_group: group object which current is in
        :return: None
        """
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
        :param contour_group: same format as output of self.find_text_contour_hierachy
        :return: list of tuples (img, point) where image is binary deskewed image where features are black,
        with width, height of self.output_width and self.output_height, point which is center base point coordinate,
        relative to ROI region, of reference line. PLEASE NOTE FEATURES ARE BLACK
        """
        return_list = []
        for group in contour_group:
            representation_contour = group[1]

            sorted_points = sorted(representation_contour[0], key=lambda x: x[0])
            r_centroid = (int(representation_contour[1][0]), int(representation_contour[1][1]))
            r_1 = sorted_points[0]
            r_2 = r_centroid

            if self.distance(r_1, r_2) <= 5:
                # Don't include contours which are too small
                continue

            angle = math.atan2(r_1[1] - r_2[1], r_1[0] - r_2[0]) * 180.0 / math.pi

            angle1 = (angle + 360) % 360
            angle2 = (angle1 + 180) % 360
            t_img = cv2.bitwise_not(input_img)  # now features/contours are black on white background

            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
            # Remove baseline contour by filling it to avoid feeeding baseline with the character in image
            cv2.drawContours(t_img, [representation_contour[3]], -1, (255, 255, 255), -1)
            t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)

            rows, cols = t_img.shape
            root_mat = cv2.getRotationMatrix2D(r_centroid, angle, 1)
            rotated = cv2.warpAffine(t_img, root_mat, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

            cropped_result = rotated[
                             max(int(r_centroid[1] - self.output_height), 0):min(int(r_centroid[1]), rotated.shape[0]),
                             max(0, int(r_centroid[0] - self.output_width)):min(rotated.shape[1],
                                                                      int(r_centroid[0] + self.output_width))]

            root_mat_2 = cv2.getRotationMatrix2D(r_centroid, angle2, 1)
            rotated_2 = cv2.warpAffine(t_img, root_mat_2, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

            cropped_result_2 = rotated_2[
                             max(int(r_centroid[1] - self.output_height), 0):min(int(r_centroid[1]),
                                                                                 rotated.shape[0]),
                             max(0, int(r_centroid[0] - self.output_width)):min(rotated.shape[1],
                                                                                int(r_centroid[0] + self.output_width))]

            # Compare image with normal angle and inverted angle and select angle which yields more black
            if cv2.countNonZero(cropped_result) > cv2.countNonZero(cropped_result_2):
                cropped_result = cropped_result_2

            # IMPORTANT: This post-processing process was added since it increased accuracy.
            #cv2.imwrite("images/cropped_letters/pre-%s.png"%(str(r_centroid)), cropped_result)
            #_, after_processing = cv2.threshold(cropped_result, 254 , 255, cv2.THRESH_BINARY)
            #final_img = cv2.erode(cropped_result, np.ones((3, 3), dtype=np.uint8))
            #final_img = after_processing
            return_list.append((cropped_result, r_centroid))

        return return_list

    def fast_locate_text(self, input_img):
        """
        On an average PC, total processing time for extracting 4 characters and running tesseract requires about 0.8
        seconds. Since the text in-game moves horizontally real time, we need to be able to get near-realtime
        coordinates of text features, compare the x values with the solved text and use the new coordinates.
        This function rapidly returns the approximate x, y coordinates of contours found on the image.
        :param input_img: full input image of gersang window.
        :return: A list of tuples (x,y) where each tuple is on top of a significant feature detected.
        """
        pass

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
        pass

if __name__ == "__main__":
    prc = Preprocessor()
    time_avg = 0
    image_count = 0
    for x in range(2, 12):
        print("-" * 20)
        print(x)
        img = prc.crop_roi(cv2.imread("images/source/%d.png"%(x), cv2.IMREAD_COLOR))
        representation = img.copy()
        start_time = time.time()
        bg_removal_start = time.time()
        tx = prc.remove_background(img)
        bg_removal_time = time.time() - bg_removal_start
        process_start = time.time()
        processed = prc.preprocess_alt(tx)
        processing_time = time.time()-process_start
        #cv2.drawContours(tx, cont, -1, (0,255,0), -1)
        #cv2.imshow("processed", processed)
        #cv2.imshow("bg", tx)
        contour_start = time.time()
        contour_groups = prc.find_text_contour_hierachy(processed)
        clustering_time = time.time() - contour_start
        for group in contour_groups:
            color = (random.randrange(0, 254), random.randrange(0, 254), random.randrange(0, 254))
            #print(group[0])
            #print(group[1])
            contours, lowest_contour = group
            #print(lowest_contour)
            cv2.circle(representation, lowest_contour[1], 3, (255,0,0), -1)

            cv2.circle(representation, lowest_contour[0][0], 2, (0,255,0), -1)
            #cv2.circle(representation, lowest_contour[0][1], 2, (0,0,255), -1)



        deskew_start = time.time()
        deskewed = prc.find_baseline_and_deskew_from_contour(processed, contour_groups)
        deskew_time = time.time() - deskew_start
        for img, ct in deskewed:
            cv2.imwrite("images/cropped_letters/%d-%s.png"%(x, str(ct)), img)
        """ocr_start = time.time()
        for img, ct in deskewed:
            prc.run_tesseract(img)
        ocr_time = time.time() - ocr_start"""
        #cv2.imshow("rep", representation)
        total_time = time.time() - start_time
        time_avg = total_time if not time_avg else (time_avg*image_count+total_time)/(image_count+1)
        image_count += 1

        print("bg removal:".ljust(20, " "), str(round(bg_removal_time, 4)) + "s",round(bg_removal_time / total_time * 100, 1), "%")
        print("preprocessing:".ljust(20, " "), str(round(processing_time, 4))+"s", round(processing_time/total_time*100, 1), "%")
        print("clustering:".ljust(20, " "), str(round(clustering_time, 4))+"s", round(clustering_time / total_time * 100, 1), "%")
        print("deskewing:".ljust(20, " "), str(round(deskew_time, 4))+"s", round(deskew_time / total_time * 100, 1), "%")
        #print("OCR:".ljust(20, " "), str(round(ocr_time, 4)) + "s", round(ocr_time / total_time * 100, 1),"%")
        print("")
        print("total time:".ljust(20, " "), str(round(total_time, 4)) + "s")
        #cv2.waitKey(0)

    print("-" * 16)
    print("average total time:".ljust(20, " "), str(round(time_avg, 4))+"s")
    cv2.destroyAllWindows()






