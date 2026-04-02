import logging
import threading

import cv2
import numpy as np


class MotionTrigger(object):
    def __init__(self, trigger_queue: threading.Semaphore, busy_event: threading.Event):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=True)
        self.trigger_queue = trigger_queue
        self.busy_event = busy_event

        self.logger = logging.getLogger()

    def motion_detection(self, frame, min_area=1200):
        if self.busy_event.is_set():
            return

        cv2.waitKey(30)
        fgmask = self.fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask.copy(), 180, 255, cv2.THRESH_BINARY)
        # creating a kernel of 4*4
        kernel = np.ones((7, 7), np.uint8)
        # applying errosion to avoid any small motion in video
        thresh = cv2.erode(thresh, kernel)
        # dilating our image
        thresh = cv2.dilate(thresh, np.ones((3, 3)), iterations=6)

        # finding the contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                if self.trigger_queue._value == 0:
                    self.trigger_queue.release()
                    self.logger.info("Motion trigger accepted")
                else:
                    self.logger.info("GPIO trigger ignored (YOLO busy)")
                return

    def cleanup(self):
        pass
