import logging
import sys
import threading
import time

import cv2

from webcam_ai.motion_trigger import MotionTrigger


class Camera:
    def __init__(self, source: str, resolution: str = "640x480"):
        self.source = source
        self.resolution = resolution
        self.resize = False

        self.logger = logging.getLogger()
        self.frame_source()

    def frame_source(self):
        """Parse input and initialize the specific camera backend."""
        if "usb" in self.source:
            self.source_type = "usb"
            try:
                idx = int(self.source.replace("usb", ""))
                self.capture = cv2.VideoCapture(idx)
                # Optimization: Set hardware-level resolution if possible
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resW)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resH)
            except ValueError:
                self.logger.error(f"Invalid USB index in {self.source}")
                sys.exit(1)

        elif "picamera" in self.source:
            try:
                import libcamera
                from picamera2 import Picamera2

                self.source_type = "picamera"
                self.capture = Picamera2()
                # Use main/lores as needed; here we align with your original lores logic
                config = self.capture.create_video_configuration(
                    main={"size": (1920, 1080), "format": "RGB888"},
                    lores={"size": (self.resW, self.resH), "format": "RGB888"},
                    transform=libcamera.Transform(hflip=True, vflip=True),
                )
                self.capture.configure(config)
                self.capture.start()
            except ImportError:
                self.logger.error("PiCamera2 or libcamera not found.")
                sys.exit(1)
        else:
            self.logger.error(f"Unsupported source: {self.source}")
            sys.exit(1)

    @property
    def resolution(self) -> str:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: str):
        self.resize = True
        self.resW, self.resH = int(resolution.split("x")[0]), int(
            resolution.split("x")[1]
        )
        self._resolution = resolution

    def read(self):
        if self.source_type == "usb":
            ret, frame = self.capture.read()

        elif self.source_type == "picamera":
            # Picamera2 capture_array is usually blocking and returns the array
            frame = self.capture.capture_array("lores")
            ret = frame is not None

        if not ret or frame is None:
            self.logger.error("Failed to retrieve frame. Camera may be disconnected.")
            return False, None

        # Resize if the hardware output doesn't match the requested internal resolution
        if self.resize:
            # Check if frame size already matches to avoid redundant compute
            if frame.shape[1] != self.resW or frame.shape[0] != self.resH:
                frame = cv2.resize(
                    frame, (self.resW, self.resH), interpolation=cv2.INTER_LINEAR
                )

        return True, frame

    def release(self):
        self.logger.info("Shutting down camera")
        if self.source_type == "usb":
            self.capture.release()
        elif self.source_type == "picamera":
            self.capture.stop()

    def __del__(self):
        self.release()


class StreamState:
    def __init__(self):
        self.latest_jpeg = None


class CameraWorker:
    def __init__(
        self, camera: Camera, stream_state: StreamState, stop_event: threading.Event
    ):
        self.camera = camera
        self.state = stream_state
        self.stop_event = stop_event
        self.logger = logging.getLogger()

    def run(self):
        try:
            while not self.stop_event.is_set():
                ret, frame = self.camera.read()
                if ret:
                    # Keep only the latest frame in the queue to prevent lag
                    _, buffer = cv2.imencode(".jpg", frame)  # type: ignore
                    self.state.latest_jpeg = buffer.tobytes()
                else:
                    time.sleep(0.1)  # Wait if camera is struggling

                # Tiny sleep to let other threads work
                time.sleep(0.01)
        finally:
            logging.info("Releasing camera...")
            self.camera.release()
