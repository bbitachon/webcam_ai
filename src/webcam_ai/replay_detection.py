import os
import queue
import threading
from datetime import datetime, timedelta

import click

from webcam_ai.detection_worker import YOLOWorker  # Change this to your actual filename


@click.group()
def cli():
    """Tool to reprocess past video events through YOLO."""
    pass


def get_event_files(directory="logging"):
    """Helper to list and parse existing event files."""
    files = []
    for f in os.listdir(directory):
        if f.startswith("event_") and f.endswith(".mp4"):
            # Extract timestamp from filename: event_2026-04-03_101849.mp4
            try:
                ts_str = f.replace("event_", "").replace(".mp4", "")
                dt = datetime.strptime(ts_str, "%Y-%m-%d_%H%M%S")
                files.append(
                    {
                        "path": os.path.join(directory, f),
                        "filename": f,
                        "datetime": dt,
                        "timestamp": ts_str,
                    }
                )
            except ValueError:
                continue
    # Sort by date descending (newest first)
    return sorted(files, key=lambda x: x["datetime"])


@cli.command()
@click.option("--n", default=5, help="Number of most recent events to process.")
@click.option(
    "--model",
    default="./train11/weights/best_ncnn_model",
    help="Path to YOLO model.",
)
def last(n, model):
    """Process the last N events."""
    events = get_event_files()[:n]
    if not events:
        click.echo("No events found.")
        return

    click.echo(f"Queuing the last {len(events)} events...")
    run_worker_on_events(events, model)


@cli.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True, help="End date (YYYY-MM-DD).")
@click.option(
    "--model", default="./train11/weights/best_ncnn_model", help="Path to YOLO model."
)
def range(start, end, model):
    """Process events within a specific date range."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    # Set end_dt to the end of that day
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)

    all_events = get_event_files()
    filtered = [e for e in all_events if start_dt <= e["datetime"] < end_dt]

    if not filtered:
        click.echo(f"No events found between {start} and {end}.")
        return

    click.echo(f"Queuing {len(filtered)} events from range...")
    run_worker_on_events(filtered, model)


def run_worker_on_events(events, model_path):
    # Setup shared resources for the worker
    detection_queue = queue.Queue()
    behavior_queue = queue.Queue()
    stop_event = threading.Event()
    busy_event = threading.Event()  # Ensure this is NOT set
    last_active = {"time": datetime(1970, 1, 1)}  # Set to old date to bypass idle check

    # Load events into the queue
    for e in events:
        detection_queue.put(
            {
                "event_dir": e["path"],
                "timestamp": e["timestamp"],
                "timestamp_iso": e["datetime"].isoformat(),
            }
        )

    # Initialize the Worker
    worker = YOLOWorker(
        model=model_path,
        detection_queue=detection_queue,
        behavior_queue=behavior_queue,
        stop_event=stop_event,
        busy_event=busy_event,
        last_active_time=last_active,
        idle_seconds=0,  # Force immediate processing
    )

    # Process until queue is empty
    click.echo("Starting YOLO Worker. Press Ctrl+C to stop.")
    try:
        while not detection_queue.empty():
            task = detection_queue.get()
            worker.process_event(
                task["event_dir"], task["timestamp"], task["timestamp_iso"]
            )
            detection_queue.task_done()
            click.echo(f"Finished: {task['event_dir']}")
    except KeyboardInterrupt:
        click.echo("\nStopping...")


if __name__ == "__main__":
    cli()
