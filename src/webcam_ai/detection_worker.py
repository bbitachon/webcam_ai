import logging
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class YOLOWorker(object):
    def __init__(
        self,
        model: str,
        detection_queue: queue.Queue,
        behavior_queue: queue.Queue,
        stop_event: threading.Event,
        busy_event: threading.Event,
        last_active_time,
        idle_seconds: int,
    ):
        self.detection_queue = detection_queue
        self.behavior_queue = behavior_queue
        self.stop_event = stop_event
        self.busy_event = busy_event
        self.last_active_time = last_active_time
        self.idle_seconds = idle_seconds
        self.logger = logging.getLogger()
        self.model = model
        self.save_dir = "logging"

        self.no_detection_dir = os.path.join(self.save_dir, "no_detections")
        os.makedirs(self.no_detection_dir, exist_ok=True)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str):
        if not os.path.exists(model):
            self.logger.error(
                "ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly."
            )
            sys.exit(0)

        self._implemented_model = YOLO(model, task="detect")
        self.labels = self._implemented_model.names
        self._model = model

    def append_log(self, filename, rows):

        df = pd.DataFrame(rows)

        log_path = os.path.join(self.save_dir, filename)

        os.makedirs(self.save_dir, exist_ok=True)

        file_exists = os.path.isfile(log_path)

        df.to_csv(
            log_path,
            mode="a",
            index=False,
            header=not file_exists,
        )

    def run(self):

        while not self.stop_event.is_set():

            if self.busy_event.is_set():
                time.sleep(1)
                continue

            now = datetime.now()
            idle_time = (now - self.last_active_time["time"]).total_seconds()

            if idle_time < self.idle_seconds:
                time.sleep(60 * 10)
                continue

            if self.detection_queue.empty():
                time.sleep(1)
                continue

            task = self.detection_queue.get()

            event_dir = task["event_dir"]
            timestamp = task["timestamp"]
            timestamp_iso = task["timestamp_iso"]

            self.logger.info("Running YOLO detection on event frames")

            self.process_event(event_dir, timestamp, timestamp_iso)
            self.detection_queue.task_done()

    def process_event(self, event_dir, timestamp, timestamp_iso):
        self.logger.info(f"Processing event: {event_dir}")

        counts = {}
        conf_sums = {}

        # 1. Run YOLO directly on the file path
        # stream=True makes it a generator (saves memory on your Pi 5)
        # conf=0.5 ignores weak detections
        results = self._implemented_model.predict(
            source=event_dir, stream=True, verbose=False
        )

        for result in results:

            detections = result.boxes

            if detections is None or len(detections) == 0:
                continue

            for box in detections:

                conf = box.conf.item()
                classidx = int(box.cls.item())
                classname = self.labels[classidx]

                if conf > 0.66:
                    counts[classname] = counts.get(classname, 0) + 1
                    conf_sums[classname] = conf_sums.get(classname, 0) + conf

        if not counts:
            self.logger.info("No detections found by YOLO")
            try:
                # Construct the destination path
                file_name = os.path.basename(event_dir)
                dest_path = os.path.join(self.no_detection_dir, file_name)

                # Move the file
                shutil.move(event_dir, dest_path)
            except Exception as e:
                self.logger.error(f"Failed to move file: {e}")
            return

        rows = []

        for cls_name in counts:

            total = counts[cls_name]

            avg_conf = round(conf_sums[cls_name] / total, 2)

            rows.append(
                {
                    "timestamp iso": timestamp_iso,
                    "timestamp": timestamp,
                    "class": cls_name,
                    "count": total,
                    "total_confidence": conf_sums[cls_name],
                    "avg_confidence": avg_conf,
                }
            )

        self.append_log("detection_log.csv", rows)

        self.behavior_queue.put(
            {
                "event_dir": event_dir,
                "timestamp": timestamp,
                "timestamp_iso": timestamp_iso,
            }
        )
