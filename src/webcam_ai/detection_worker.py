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

    def save_timeline(self, event_dir, timeline_data, timestamp_iso):
        """Saves detailed frame-by-frame detections to a CSV."""
        timeline_dir = os.path.join(self.save_dir, "detection_timelines")
        os.makedirs(timeline_dir, exist_ok=True)

        # 1. Extract just the file name (e.g., "video.mp4" instead of "/dir/video.mp4")
        file_only = os.path.basename(event_dir)

        # 2. Remove the extension
        base_name = os.path.splitext(file_only)[0]

        # 3. Add suffix to avoid overwriting the behavior timeline
        csv_path = os.path.join(timeline_dir, f"{base_name}.csv")

        # 4. Build the DataFrame from our frame dictionaries
        df = pd.DataFrame(timeline_data)

        # Ensure columns are ordered: frame, timestamp_iso, then the cat names
        cols = ["frame", "timestamp_iso"] + list(self.labels.values())
        df = df.reindex(columns=cols, fill_value=0.0)

        df.to_csv(csv_path, index=False)
        self.logger.debug(f"Timeline saved to {csv_path}")

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
                time.sleep(60 * 5)
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
        timeline_data = []

        # 1. Run YOLO directly on the file path
        # stream=True makes it a generator (saves memory on your Pi 5)
        # conf=0.5 ignores weak detections
        results = self._implemented_model.predict(
            source=event_dir, stream=True, verbose=False
        )

        for frame_idx, result in enumerate(results):

            frame_row = {"frame": frame_idx, "timestamp_iso": timestamp_iso}
            for cls_name in self.labels.values():
                frame_row[cls_name] = 0

            detections = result.boxes

            if detections is None or len(detections) == 0:
                continue

            for box in detections:

                conf = box.conf.item()
                classidx = int(box.cls.item())
                classname = self.labels[classidx]

                # Record the max confidence for this class in this frame
                if conf > frame_row[classname]:
                    frame_row[classname] = round(conf, 3)

                if conf > 0.66:
                    counts[classname] = counts.get(classname, 0) + 1
                    conf_sums[classname] = conf_sums.get(classname, 0) + conf

            timeline_data.append(frame_row)

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

        if timeline_data:
            self.save_timeline(event_dir, timeline_data, timestamp_iso)

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
