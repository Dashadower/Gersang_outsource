import logging, os
logger = logging.getLogger("log")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
fh = logging.FileHandler("logging.log", encoding="utf-8")
logger.addHandler(fh)
logger.debug("Program start")
try:
    from image_processor import Preprocessor
    from screencap import ScreenCapturer
    import cv2, imutils, numpy as np, time
except:
    logger.exception("오류 발생!")
    os.system("pause")
scap = ScreenCapturer()
window_handle = scap.ms_get_screen_hwnd()
processor = Preprocessor(background_img="background.png")
logger.debug("Window handle: " + str(window_handle))
if not window_handle:
    logging.debug("No gersang window found.")
    print("거상창을 찾지 못했습니다.")

else:
    try:
        while True:
            
            captured_img = cv2.cvtColor(np.array(scap.capture(set_focus=False, hwnd=window_handle), dtype=np.uint8), cv2.COLOR_RGB2BGR)

            cropped = processor.crop_roi(captured_img)
            render = cropped.copy()
            cv2.imshow("window", imutils.resize(cropped, width=300))
            inp = cv2.waitKey(1)
            if inp == ord("q"):
                representation = captured_img.copy()
                start = time.time()

                bg_removed = processor.remove_background(cropped)
                processed = processor.preprocess_alt(bg_removed)
                contour_groups = processor.find_text_contour_hierachy(processed)
                for deskewed, ct in processor.find_baseline_and_deskew_from_contour(processed, contour_groups):
                    result = processor.run_tesseract(deskewed)

                    if result.isalpha() or result.isdigit():
                        cv2.putText(representation, result[0], ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    else:
                        cv2.putText(representation, result, ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                print("처리시간:", time.time() - start)
                cv2.imshow("window", representation)
                cv2.waitKey(0)

    except:
        logging.exception("EXCEPTION")


