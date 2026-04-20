import logging
import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime

import cv2
import ncnn
import numpy as np
import pandas as pd
from ultralytics import YOLO


class BaseWorker(ABC):

    @abstractmethod
    def __init__(
        self,
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

        self.last_active_time = last_active_time
        self.idle_seconds = idle_seconds

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

    @abstractmethod
    def process_event(self, event_dir, timestamp, timestamp_iso):
        pass


class BehaviorWorker(BaseWorker):

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
        super().__init__(
            detection_queue=detection_queue,
            behavior_queue=behavior_queue,
            busy_event=busy_event,
            stop_event=stop_event,
            last_active_time=last_active_time,
            idle_seconds=idle_seconds,
        )

        self.squat_model = squat_model
        self.pee_model = pee_model

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
        self.logger.info(f"Finished processing behavior event: {event_dir}")


class BehaviorWorker_x3d(BaseWorker):
    def __init__(
        self,
        model: str,
        detection_queue: queue.Queue,
        behavior_queue: queue.Queue,
        busy_event: threading.Event,
        stop_event: threading.Event,
        last_active_time,
        idle_seconds: int,
    ):
        super().__init__(
            detection_queue=detection_queue,
            behavior_queue=behavior_queue,
            busy_event=busy_event,
            stop_event=stop_event,
            last_active_time=last_active_time,
            idle_seconds=idle_seconds,
        )
        self.model = model
        self._image_size = 160
        self._clip_length = 30

    def save_timeline(self, video_name, full_probs, timestamp_iso):
        """Saves detailed frame-by-frame probabilities to a CSV."""
        timeline_dir = os.path.join("logging", "timelines")
        os.makedirs(timeline_dir, exist_ok=True)

        # 1. Extract just the file name (e.g., "video.mp4" instead of "/dir/video.mp4")
        file_only = os.path.basename(video_name)

        # 2. Remove the extension
        base_name = os.path.splitext(file_only)[0]

        csv_path = os.path.join(timeline_dir, f"{base_name}_timeline.csv")

        df = pd.DataFrame(full_probs, columns=["idle", "peeing", "pooing"])
        df.insert(0, "frame", range(len(df)))
        df.insert(0, "timestamp_iso", timestamp_iso)

        df.to_csv(csv_path, index=False)
        self.logger.debug(f"Timeline saved to {csv_path}")

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model_path: str):

        self._init_ncnn_model(model_path)

        self.labels = {0: "idle", 1: "peeing", 2: "pooing"}
        self._mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self._std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    def _init_ncnn_model(self, model_path: str):
        if not os.path.isdir(model_path):
            self.logger.error(f"Provided path is not a directory: {model_path}")
            sys.exit(0)

        # Search for .bin and .param files in the directory
        bin_files = [f for f in os.listdir(model_path) if f.endswith(".bin")]
        param_files = [f for f in os.listdir(model_path) if f.endswith(".param")]

        if not bin_files or not param_files:
            self.logger.error(f"No .bin or .param files found in {model_path}")
            sys.exit(0)

        # Match the first bin file found
        # (Assuming the bin and param share the same base name)
        bin_path = os.path.join(model_path, bin_files[0])
        param_name = bin_files[0].replace(".bin", ".param")
        param_path = os.path.join(model_path, param_name)

        if not os.path.exists(param_path):
            self.logger.error(f"Found {bin_files[0]} but no matching {param_name}")
            sys.exit(0)

        # Initialize NCNN
        self.net = ncnn.Net()

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self._model = bin_path  # Store the bin path for reference
        self.logger.info(f"Successfully loaded NCNN model from: {model_path}")

    def preprocess_frame(self, frame):
        """Resizes, converts to RGB, and normalizes a single frame."""
        frame = cv2.resize(frame, (self._image_size, self._image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - self._mean) / self._std
        return frame

    def interpolate_probs(self, results, total_frames):
        """Interpolates sparse clip probabilities over the full frame timeline."""
        frame_idxs = results[:, 0]
        probs = results[:, 1:]

        full_probs = np.zeros((total_frames, probs.shape[1]))

        for i in range(probs.shape[1]):
            full_probs[:, i] = np.interp(
                np.arange(total_frames), frame_idxs, probs[:, i]
            )

        return full_probs

    def smooth_probs(self, results, k=3):
        """Applies a moving average to smooth probabilities."""
        smoothed = results.copy()
        for i in range(len(results)):
            start = max(0, i - k)
            end = min(len(results), i + k + 1)
            smoothed[i, 1:] = np.mean(results[start:end, 1:], axis=0)
        return smoothed

    def run_inference(self, video_path: str, stride: int = 2, batch_size: int = 4):
        """
        Extracts sliding window clips from the video and runs NCNN inference.
        """
        cap = cv2.VideoCapture(video_path)
        all_frames = []

        # 1. Load and preprocess all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(self.preprocess_frame(frame))
        cap.release()

        total_frames = len(all_frames)
        if total_frames < self._clip_length:
            return np.array([]), total_frames

        # 2. Prepare sliding windows
        windows = []
        mid_frames = []

        for start_idx in range(0, total_frames - self._clip_length + 1, stride):
            window = all_frames[start_idx : start_idx + self._clip_length]
            windows.append(window)
            mid_frames.append(start_idx + (self._clip_length // 2))

        results = []

        # 3. Process Windows (Clip Batching logic, executed sequentially for NCNN)
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_mids = mid_frames[i : i + batch_size]

            for j, window in enumerate(batch_windows):
                # Convert to numpy array: shape (T, H, W, C)
                input_np = np.array(window)

                # Transpose to PNNX/NCNN format: (C, T, H, W)
                input_np = np.transpose(input_np, (3, 0, 1, 2))
                input_np = np.ascontiguousarray(input_np)

                # Create NCNN Mat
                mat = ncnn.Mat(input_np)

                # Run Extractor
                ex = self.net.create_extractor()
                ex.input("in0", mat)  # "in0" is standard PNNX input name
                _, out = ex.extract("out0")  # "out0" is standard PNNX output name

                probs = np.array(out)
                results.append([batch_mids[j]] + probs.tolist())

        return np.array(results), total_frames

    def process_event(self, event_dir, timestamp, timestamp_iso):
        """Processes the video event using the X3D sliding window pipeline."""
        self.logger.info(f"Processing behavior event: {event_dir}")

        # Locate the video file inside the event_dir (if it's a directory)
        video_path = event_dir
        if os.path.isdir(event_dir):
            video_files = [
                f
                for f in os.listdir(event_dir)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
            if not video_files:
                self.logger.warning(f"No video file found in {event_dir}")
                return
            video_path = os.path.join(event_dir, video_files[0])

        probs_over_time, total_frames = self.run_inference(
            video_path, stride=2, batch_size=8
        )

        if total_frames == 0 or len(probs_over_time) == 0:
            self.logger.info("Video too short or empty - skipping behavior analysis")
            return

        # 1. Interpolate to full frame resolution
        full_probs = self.interpolate_probs(probs_over_time, total_frames)

        # 2. Smooth probabilities
        temp_results = np.column_stack([np.arange(len(full_probs)), full_probs])
        full_probs = self.smooth_probs(temp_results)[:, 1:]

        self.save_timeline(video_path, full_probs, timestamp_iso)

        # 3. Compute areas (Total Confidence for the UI)
        areas = np.sum(full_probs, axis=0)

        # Build rows for the CSV that matches the old YOLO format so UI works perfectly
        rows = []
        for class_idx, class_name in self.labels.items():
            area_score = round(float(areas[class_idx]), 2)

            # Only log if the area score is meaningful (filters out static noise)
            if area_score > 1.0:
                rows.append(
                    {
                        "timestamp iso": timestamp_iso,
                        "timestamp": timestamp,
                        "class": class_name,
                        "total_confidence": area_score,
                    }
                )

        if not rows:
            self.logger.info("No dominant behavior detected.")
            return

        # Save to video_log.csv so main.py Plotly graphs pick it up automatically
        self.append_log("video_log.csv", rows)
        self.logger.info(f"Finished processing behavior event: {event_dir}")


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
