import logging
import sys



class Camera:
    def __init__(self, source: str, resolution: str = "640x480"):
        self.source = source
        self.frame_source()
        self.resize = False
        self.resolution = resolution
        self.logger = logging.getLogger()

    def frame_source(self):
        # Parse input to determine if image source is a file, folder, video, or USB camera

        if "usb" in self.source:
            import cv2
            self.source_type = "usb"
            usb_idx = int(self.source[3:])
        elif "picamera" in self.source:
            import libcamera
            from picamera2 import Picamera2
            self.source_type = "picamera"
            picam_idx = int(self.source[8:])
        else:
            self.logger.error(f"Input {self.source} is invalid. Please try again.")
            sys.exit(0)

        # Load or initialize image source
        if self.source_type == "usb":
            cap_arg = usb_idx
            self.capture = cv2.VideoCapture(cap_arg)

            # Set camera or video resolution if specified by user
            # if resolution:
            #     _ = cap.set(3, resW)
            #     _ = cap.set(4, res
            # H)

        elif self.source_type == "picamera":

            self.capture = Picamera2()
            self.capture.configure(
                self.capture.create_video_configuration(
                    main={"size": (1920, 1080), "format": "RGB888"},
                    lores={"size": (640, 480), "format": "RGB888"},
                    transform=libcamera.Transform(hflip=True, vflip=True),
                )
            )
            self.capture.start()

    @property
    def resolution(self) -> str:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: str):
        self.resize = False
        self.resW, self.resH = int(resolution.split("x")[0]), int(
            resolution.split("x")[1]
        )
        self._resolution = resolution

    def read(self):
        if (
            self.source_type == "usb"
        ):  # If source is a USB camera, grab frame from camera
            ret, frame = self.capture.read()
            if (frame is None) or (not ret):
                self.logger.error(
                    "Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program."
                )
        elif (
            self.source_type == "picamera"
        ):  # If source is a Picamera, grab frames using picamera interface
            frame = self.capture.capture_array("lores")
            ret = True
            if frame is None:
                self.logger.error(
                    "Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program."
                )
                ret = False

        # Resize frame to desired display resolution
        if ret == True:

            if self.resize == True:
                frame = cv2.resize(
                    frame, (self.resW, self.resH), interpolation=cv2.INTER_LINEAR
                )
        return ret, frame

    def release(self):
        self.logger.info("Shutting down camera")
        if self.source_type == "usb":
            self.capture.release()
        elif self.source_type == "picamera