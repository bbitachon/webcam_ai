import os
import queue
import threading
from datetime import datetime, timedelta

import click

# Import your BehaviorWorker class
from webcam_ai.behavioral_worker import BehaviorWorker


@click.group()
def cli():
    """Tool to reprocess past video events through Behavioral Analysis (Squat/Pee)."""
    pass


def get_event_files(directory="logging"):
    """Helper to list and parse existing event files from the folder."""
    files = []
    if not os.path.exists(directory):
        return []

    for f in os.listdir(directory):
        if f.startswith("event_") and f.endswith(".mp4"):
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
    # Sort by date ascending (oldest first)
    return sorted(files, key=lambda x: x["datetime"])


@cli.command()
@click.option("--n", default=5, help="Number of most recent events to process.")
@click.option("--squat-model", default="squatting_model/weights/best_ncnn_model")
@click.option("--pee-model", default="peeing_model/weights/best_ncnn_model")
def last(n, squat_model, pee_model):
    """Process the last N events found in the logging folder."""
    events = get_event_files()[:n]
    if not events:
        click.echo("No event files found.")
        return

    click.echo(f"Processing the last {len(events)} events...")
    run_behavior_on_events(events, squat_model, pee_model)


@cli.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", required=True, help="End date (YYYY-MM-DD).")
@click.option("--squat-model", default="squatting_model/weights/best_ncnn_model")
@click.option("--pee-model", default="peeing_model/weights/best_ncnn_model")
def range(start, end, squat_model, pee_model):
    """Process events within a specific date range."""
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        click.echo("Error: Date format must be YYYY-MM-DD")
        return

    all_events = get_event_files()
    filtered = [e for e in all_events if start_dt <= e["datetime"] < end_dt]

    if not filtered:
        click.echo(f"No events found between {start} and {end}.")
        return

    click.echo(f"Processing {len(filtered)} events from range...")
    run_behavior_on_events(filtered, squat_model, pee_model)


def run_behavior_on_events(events, squat_path, pee_path):
    # Setup shared resources to satisfy BehaviorWorker's __init__
    detection_queue = queue.Queue()
    behavior_queue = queue.Queue()
    stop_event = threading.Event()
    busy_event = threading.Event()  # Keep clear so worker doesn't pause
    # Set last_active far in the past to bypass idle_seconds check
    last_active = {"time": datetime.now() - timedelta(days=365)}

    # Initialize the Behavior Worker
    worker = BehaviorWorker(
        squat_model=squat_path,
        pee_model=pee_path,
        detection_queue=detection_queue,
        behavior_queue=behavior_queue,
        busy_event=busy_event,
        stop_event=stop_event,
        last_active_time=last_active,
        idle_seconds=0,  # No cooldown needed for replay
    )

    click.echo("Starting Behavior Analysis. Press Ctrl+C to stop.")
    try:
        for e in events:
            # We call process_event directly on each file
            click.echo(f"Analyzing: {e['filename']}...")

            # Using the timestamp from the filename ensures logs match the original event
            worker.process_event(e["path"], e["timestamp"], e["datetime"].isoformat())

    except KeyboardInterrupt:
        click.echo("\nStopping...")


if __name__ == "__main__":
    cli()
