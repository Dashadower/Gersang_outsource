import logging, os
logger = logging.getLogger("log")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
fh = logging.FileHandler("logging.log", encoding="utf-8")
logger.addHandler(fh)
logger.debug("Program start")
try:
    from image_processor import Preprocessor
    from screencap import MapleScreenCapturer
    import cv2
except:
    logger.exception("오류 발생!")
    os.system("pause")
scap = MapleScreenCapturer()
window_handle = scap.ms_get_screen_hwnd()
processor = Preprocessor(background_img="background.png")
logger.debug("Window handle: " + str(window_handle))
if not window_handle:
    print("거상창을 찾지 못했습니다.")
    os.system("pause")
else:
    try:
        while True:
            captured_img = scap.capture(set_focus=False, hwnd=window_handle)
            cropped = processor.crop_roi()
            cv2.imshow("window", cropped)
            inp = cv2.waitKey(1)
            if inp == ord("q"):
                bg_removed = processor.remove_background(cropped)
                cv2.imwrite("bg_removed.png", bg_removed)
                processed = processor.threshold_and_preprocess(bg_removed)
                cv2.imwrite("processed", processed)
                contour_groups = processor.find_text_contour_hierachy(processed)
                print(contour_groups)
                render = cropped.copy()
                for deskewed, ct in processor.find_baseline_and_deskew_from_contour(processed, contour_groups):

                    result = processor.run_tesseract(deskewed)
                    if result.isalpha():
                        result = result.upper()
                    if len(result) > 1:
                        result = "?"
                    print(ct)
                    cv2.putText(render, result, ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("window", render)
                cv2.waitKey(0)

    except:
        logging.exception("EXCEPTION")
        print("오류 발생!")
        os.system("pause")
