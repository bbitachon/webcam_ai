import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime

import pandas as pd
from ultralytics import YOLO


class BehaviorWorker(object):

    def __init__(
        self,
        squat_model: str,
        pee_model: str,
        detection_queue: queue.Queue,
        behavior_queue: queue.Queue,
        busy_event: threading.Event,
        stop_event: threading.Event,
        last_active_time,
        idle_seconds: int,
    ):
        self.behavior_queue = behavior_queue
        self.detection_queue = detection_queue
        self.busy_event = busy_event
        self.stop_event = stop_event

        self.logger = logging.getLogger()

        self.squat_model = squat_model
        self.pee_model = pee_model
        self.last_active_time = last_active_time
        self.idle_seconds = idle_seconds

    @property
    def squat_model(self) -> str:
        return self._squat_model

    @squat_model.setter
    def squat_model(self, squat_model: str):
        if not os.path.exists(squat_model):
            print(
                "ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly."
            )
            sys.exit(0)

        self._implemented_squat_model = YOLO(squat_model, task="detect")
        self.squat_labels = self._implemented_squat_model.names
        self._squat_model = squat_model

    @property
    def pee_model(self) -> str:
        return self._pee_model

    @pee_model.setter
    def pee_model(self, pee_model: str):
        if not os.path.exists(pee_model):
            print(
                "ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly."
            )
            sys.exit(0)

        self._implemented_pee_model = YOLO(pee_model, task="detect")
        self.pee_labels = self._implemented_pee_model.names
        self._pee_model = pee_model

    def append_log(self, filename, rows):

        df = pd.DataFrame(rows)

        log_path = os.path.join("logging", filename)

        os.makedirs("logging", exist_ok=True)

        file_exists = os.path.isfile(log_path)

        df.to_csv(
            log_path,
            mode="a",
            index=False,
            header=not file_exists,
        )

    def run(self):

        while not self.stop_event.is_set():

            # YOLO still active
            if self.busy_event.is_set():
                time.sleep(1)
                continue

            if not self.detection_queue.empty():
                time.sleep(10)
                continue

            now = datetime.now()
            idle_duration = now - self.last_active_time["time"]

            if idle_duration.total_seconds() < self.idle_seconds:
                time.sleep(10)
                continue

            if self.behavior_queue.empty():
                time.sleep(10)
                continue

            task = self.behavior_queue.get()

            event_dir = task["event_dir"]
            timestamp = task["timestamp"]
            timestamp_iso = task["timestamp_iso"]

            self.logger.info("Running behavioral analysis")

            self.process_event(event_dir, timestamp, timestamp_iso)
            self.behavior_queue.task_done()

    def process_event(self, event_dir, timestamp, timestamp_iso):
        self.logger.info(f"Processing behavior event: {event_dir}")

        counts = {}
        conf_sums = {}

        pee_counts = {}
        pee_conf = {}

        squat_frames = []

        # -----------------------------
        # PASS 1: Squat detection
        # -----------------------------

        results = self._implemented_squat_model(event_dir, verbose=False, stream=True)

        for result in results:

            detections = result.boxes

            if detections is None:
                continue

            for box in detections:

                conf = box.conf.item()
                classidx = int(box.cls.item())
                classname = self.squat_labels[classidx]

                if conf > 0.7:
                    counts[classname] = counts.get(classname, 0) + 1
                    conf_sums[classname] = conf_sums.get(classname, 0) + conf

                    if classname == "squat":
                        squat_frames.append(result.orig_img)

        if not counts:
            return

        # -----------------------------
        # LOG SQUAT RESULT
        # -----------------------------
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

        self.append_log("squat_log.csv", rows)

        # -----------------------------
        # STOP if no squat
        # -----------------------------
        squat_score = conf_sums.get("squat", 0)
        idle_score = conf_sums.get("idle", 0)

        self.logger.info(f"Squat score={squat_score:.2f} idle score={idle_score:.2f}")

        if squat_score <= idle_score:
            self.logger.info("Idle event detected - skipping pee model")
            return

        self.logger.info("Squat dominant - running pee model")

        # -----------------------------
        # PASS 2: Pee detection
        # -----------------------------

        for frame in squat_frames:

            results = self._implemented_pee_model(frame, verbose=False, stream=True)

            for result in results:

                detections = result.boxes

                for box in detections:

                    conf = box.conf.item()
                    classidx = int(box.cls.item())
                    classname = self.pee_labels[classidx]

                    if conf > 0.7:

                        pee_counts[classname] = pee_counts.get(classname, 0) + 1
                        pee_conf[classname] = pee_conf.get(classname, 0) + conf

        if not pee_counts:
            return

        rows = []

        for cls_name in pee_counts:

            total = pee_counts[cls_name]
            avg_conf = round(pee_conf[cls_name] / total, 2)

            rows.append(
                {
                    "timestamp iso": timestamp_iso,
                    "timestamp": timestamp,
                    "class": cls_name,
                    "count": total,
                    "total_confidence": pee_conf[cls_name],
                    "avg_confidence": avg_conf,
                }
            )

        self.append_log("pee_log.csv", rows)


def main():

    # Shared objects
    behavior_queue = queue.Queue()
    detection_queue = queue.Queue()

    busy_event = threading.Event()
    stop_event = threading.Event()

    # Simulated last active time (set to past so worker runs immediately)
    last_active_time = {"time": datetime.now() - timedelta(seconds=60)}

    # Create worker
    worker = BehaviorWorker(
        squat_model="squatting_model/weights/best_ncnn_model",
        pee_model="peeing_model/weights/best_ncnn_model",
        detection_queue=detection_queue,
        behavior_queue=behavior_queue,
        busy_event=busy_event,
        stop_event=stop_event,
        last_active_time=last_active_time,
        idle_seconds=0,
    )

    # Start worker in background thread
    worker_thread = threading.Thread(target=worker.run, daemon=True)
    worker_thread.start()

    print("Behavior worker started.")

    # -----------------------------
    # CREATE TEST EVENT
    # -----------------------------
    test_event_dir = "logging/frames_2026-03-13_041858"

    if not os.path.exists(test_event_dir):
        print(f"Folder '{test_event_dir}' not found.")
        return

    task = {
        "event_dir": test_event_dir,
        "timestamp": int(time.time()),
        "timestamp_iso": datetime.now().isoformat(),
    }

    behavior_queue.put(task)

    print("Test event added to queue.")

    # Keep main alive
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping worker...")
        stop_event.set()
        worker_thread.join()


if __name__ == "__main__":
    from datetime import timedelta

    main()
