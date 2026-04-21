import logging
import os
import queue
import threading
from datetime import datetime, timedelta

import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import Response
from nicegui import app, ui
from plotly.subplots import make_subplots

# Import your class from your other file
from webcam_ai.behavioral_worker import BehaviorWorker_x3d
from webcam_ai.camera_service import Camera, CameraWorker, RecorderWorker, StreamState
from webcam_ai.detection_worker import YOLOWorker
from webcam_ai.motion_trigger import MotionTrigger

state = StreamState()


def extract_start_time(filename):
    """Extracts datetime from 'event_YYYY-MM-DD_HHMMSS.csv'"""
    try:
        # Get the 'YYYY-MM-DD_HHMMSS' part
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        date_str = f"{parts[1]}_{parts[2]}"
        return datetime.strptime(date_str, "%Y-%m-%d_%H%M%S")
    except Exception:
        return datetime.min


def load_stitched_timeline(folder, fps=10, cutoff=None, window=1):
    if not os.path.exists(folder):
        return pd.DataFrame()

    # Use your working list logic
    csv_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".csv")], key=extract_start_time
    )

    all_rows = []
    last_end = None

    for f in csv_files:
        start_time = extract_start_time(f)

        # Only process if it's within the last 24 hours
        if cutoff and start_time < cutoff:
            continue

        path = os.path.join(folder, f)
        df = pd.read_csv(path)
        if df.empty:
            continue

        # Get the columns (idle, pee, poo OR cat names)
        cols = [c for c in df.columns if c not in ["frame", "timestamp_iso"]]

        if window > 1:
            df[cols] = (
                df[cols].rolling(window=window, min_periods=1, center=True).mean()
            )

        for i in range(len(df)):
            t = start_time + timedelta(seconds=i / fps)

            # Gap handling (Your working logic)
            if last_end and (t - last_end).total_seconds() > 1:
                gap_row = {"timestamp": last_end + timedelta(milliseconds=100)}
                for c in cols:
                    gap_row[c] = None  # None breaks the line in Plotly
                all_rows.append(gap_row)

            new_row = {"timestamp": t}
            for c in cols:
                new_row[c] = df[c].iloc[i]
            all_rows.append(new_row)
            last_end = t

    return pd.DataFrame(all_rows)


# --- 1. ANALYTICS DATA LOGIC ---
def load_data():
    save_dir = "logging"
    cutoff = datetime.now() - timedelta(hours=24)

    # Detections (Kiti, Alejandro, Elsa)
    df_det = load_stitched_timeline(
        os.path.join(save_dir, "detection_timelines"), cutoff=cutoff, window=11
    )

    # Behaviors (idle, peeing, pooing)
    df_beh = load_stitched_timeline(
        os.path.join(save_dir, "behavior_timelines"), cutoff=cutoff
    )

    return df_det, df_beh


def build_figure(df_det, df_beh):
    color_map = {
        "Kiti": "red",
        "Alejandro": "blue",
        "Elsa": "green",
        "idle": "#d3d3d3",
        "peeing": "#9b59b6",
        "pooing": "#e67e22",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Detection Confidence", "Behavior Confidence"),
    )

    if not df_det.empty:
        classes = [c for c in df_det.columns if c != "timestamp"]
        for cls in classes:
            fig.add_trace(
                go.Scatter(
                    x=df_det["timestamp"],
                    y=(df_det[cls]),
                    mode="lines",
                    connectgaps=False,
                    name=f"See: {cls}",
                    line=dict(color=color_map.get(cls, "gray"), width=1.5),
                    # fill="tozeroy",
                ),
                row=1,
                col=1,
            )

    # --- Subplot 2: Behavior Timelines ---
    if not df_beh.empty:
        # Pooing/Peeing get fills to look like "events"
        for cls in ["peeing", "pooing"]:
            fig.add_trace(
                go.Scatter(
                    x=df_beh["timestamp"],
                    y=df_beh[cls],
                    name=cls.capitalize(),
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color_map[cls], width=2),
                    # fill="tozeroy",
                ),
                row=2,
                col=1,
            )

        # Idle stays as a dashed background line
        fig.add_trace(
            go.Scatter(
                x=df_beh["timestamp"],
                y=df_beh["idle"],
                name="Idle",
                mode="lines",
                connectgaps=False,
                line=dict(color=color_map["idle"], dash="dash", width=1),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    # Global Styling
    now = datetime.now()
    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # xaxis2=dict(rangeslider=dict(visible=True), type="date"),
    )

    fig.update_xaxes(range=[now - timedelta(hours=24), now], type="date")
    fig.update_yaxes(title_text="Detection Probability", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Behavior Probability", range=[0, 1.05], row=3, col=1)

    return fig


@app.get("/video/stream")
def video_stream():
    if state.latest_jpeg is not None:
        return Response(content=state.latest_jpeg, media_type="image/jpeg")
    return Response(status_code=404)


# 2. THE UI LOGIC
def start_ui(source, res, port, stop_event: threading.Event):
    @ui.page("/")
    def index():
        with ui.column().classes("w-full max-w-5xl mx-auto items-center p-4"):
            ui.label(f"Streaming: {source} ({res})").classes("text-h4")
            video_image = ui.interactive_image("/video/stream").classes(
                "w-full max-w-2xl border-2"
            )

            ui.timer(0.1, callback=video_image.force_reload)

            ui.label("Activity in the last 24 hours").classes("text-h5 mt-6")

            # Load data and build figure
            df_det, df_beh = load_data()
            fig = build_figure(df_det, df_beh)
            plotly_figure = ui.plotly(fig).classes("w-full max-w-3xl")

        async def refresh_data():
            try:
                if not ui.context.client.connected:
                    return
                if plotly_figure.client is None:
                    return

                df_det, df_beh = load_data()
                new_fig = build_figure(df_det, df_beh)
                plotly_figure.update_figure(new_fig)

            except Exception as e:
                logging.debug(f"UI update skipped: {e}")

        ui.timer(1800, callback=refresh_data)  # Refresh every 60 seconds

    # Cleanup when NiceGUI closes
    app.on_shutdown(lambda: stop_event.set())

    ui.run(port=port, title="Cat Litter Monitoring", reload=False)


# 3. THE CLI ENTRY POINT
@click.command()
@click.option("--source", default="usb0", help="usb0 or picamera0")
@click.option("--res", default="640x480", help="Resolution (WxH)")
@click.option("--port", default=8080, help="Web port")
@click.option(
    "--model", default="./train11/weights/best_ncnn_model", help="Path to YOLO model"
)
@click.option(
    "--idle-seconds", default=60 * 10, help="Idle time before running YOLO detection"
)
def main(source, res, port, model, idle_seconds):
    # Standard Python Queue (Thread-safe)
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue()
    behavior_queue = queue.Queue()
    trigger_queue = threading.Semaphore(0)
    stop_event = threading.Event()
    busy_event = threading.Event()
    last_active_time = {"time": datetime.now()}
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    camera = Camera(source=source, resolution=res)
    trigger_implemented = MotionTrigger(
        trigger_queue=trigger_queue, busy_event=busy_event
    )
    camera_worker = CameraWorker(
        camera=camera,
        frame_queue=frame_queue,
        trigger=trigger_implemented,
        stream_state=state,
        stop_event=stop_event,
    )

    recorder_worker = RecorderWorker(
        frame_queue=frame_queue,
        detection_queue=detection_queue,
        trigger_semaphore=trigger_queue,
        busy_event=busy_event,
        stop_event=stop_event,
        last_active_time=last_active_time,
    )

    yolo_worker = YOLOWorker(
        model=model,
        detection_queue=detection_queue,
        behavior_queue=behavior_queue,
        stop_event=stop_event,
        busy_event=busy_event,
        last_active_time=last_active_time,
        idle_seconds=idle_seconds,
    )

    behavior_worker = BehaviorWorker_x3d(
        model="squatting_video_model",
        detection_queue=detection_queue,
        behavior_queue=behavior_queue,
        busy_event=busy_event,
        stop_event=stop_event,
        last_active_time=last_active_time,
        idle_seconds=idle_seconds,
    )

    # Create and start the camera thread (producer)
    camera_thread = threading.Thread(
        target=camera_worker.run,
        daemon=True,  # Thread dies if the main script stops
    )
    camera_thread.start()

    # Create and start the recorder thread (consumer)
    recorder_thread = threading.Thread(
        target=recorder_worker.run,
        daemon=True,  # Thread dies if the main script stops
    )
    recorder_thread.start()

    # Create and start the YOLO detection thread
    yolo_thread = threading.Thread(
        target=yolo_worker.run,
        daemon=True,  # Thread dies if the main script stops
    )
    yolo_thread.start()

    # Create and start the Behavior analysis thread
    behavior_thread = threading.Thread(
        target=behavior_worker.run,
        daemon=True,  # Thread dies if the main script stops
    )
    behavior_thread.start()

    # Start the NiceGUI loop
    start_ui(source, res, port, stop_event)


if __name__ in {"__main__", "__mp_main__"}:
    main()
