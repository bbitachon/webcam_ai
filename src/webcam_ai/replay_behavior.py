import os
import queue
import threading
from datetime import datetime, timedelta

import click
import pandas as pd

from webcam_ai.behavioral_worker import (
    BehaviorWorker,  # Ensure this matches your filename
)


@click.group()
def cli():
    """Manually re-run Behavior Analysis on past cat detections."""
    pass


def get_cat_events_from_log(limit=None, start_date=None, end_date=None):
    """Reads detection_log.csv to find videos where a cat was actually found."""
    log_path = "logging/detection_log.csv"
    if not os.path.exists(log_path):
        click.echo("Error: detection_log.csv not found. Run YOLO replay first.")
        return []

    df = pd.read_csv(log_path)
    # Filter for 'cat' class only
    df = df[df["class"].str.lower() == "cat"]
    df["dt"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d_%H%M%S")

    if start_date and end_date:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        df = df[(df["dt"] >= s) & (df["dt"] < e)]

    df = df.sort_values("dt", ascending=False)

    if limit:
        df = df.head(limit)

    events = []
    for _, row in df.iterrows():
        # Reconstruct path from timestamp
        # Matches your RecorderWorker format: event_YYYY-MM-DD_HHMMSS.mp4
        filename = f"event_{row['timestamp']}.mp4"
        path = os.path.join("logging", filename)

        if os.path.exists(path):
            events.append(
                {
                    "event_dir": path,
                    "timestamp": row["timestamp"],
                    "timestamp_iso": row["timestamp iso"],
                }
            )
    return events


@cli.command()
@click.option("--n", default=5, help="Number of recent cat events to analyze.")
@click.option("--squat-model", default="squat_model.pt")
@click.option("--pee-model", default="pee_model.pt")
def last(n, squat_model, pee_model):
    """Analyze the last N cat detections."""
    events = get_cat_events_from_log(limit=n)
    if not events:
        click.echo("No cat events found to process.")
        return

    click.echo(f"Found {len(events)} cat videos. Starting analysis...")
    run_behavior_replay(events, squat_model, pee_model)


@cli.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--squat-model", default="squat_model.pt")
@click.option("--pee-model", default="pee_model.pt")
def range(start, end, squat_model, pee_model):
    """Analyze cat detections within a date range."""
    events = get_cat_events_from_log(start_date=start, end_date=end)
    if not events:
        click.echo(f"No cat events found between {start} and {end}.")
        return

    click.echo(f"Found {len(events)} events in range. Starting...")
    run_behavior_replay(events, squat_model, pee_model)


def run_behavior_replay(events, squat_m, pee_m):
    # Setup dummy objects to satisfy BehaviorWorker requirements
    beh_q = queue.Queue()
    det_q = queue.Queue()  # Empty detection queue so Behavior doesn't wait
    stop_event = threading.Event()
    busy_event = threading.Event()  # Ensure NOT set so worker runs
    last_act = {"time": datetime(1970, 1, 1)}

    worker = BehaviorWorker(
        squat_model=squat_m,
        pee_model=pee_m,
        detection_queue=det_q,
        behavior_queue=beh_q,
        busy_event=busy_event,
        stop_event=stop_event,
        last_active_time=last_act,
        idle_seconds=0,  # No waiting
    )

    for event in events:
        click.echo(f"Processing: {event['event_dir']}")
        worker.process_event(
            event["event_dir"], event["timestamp"], event["timestamp_iso"]
        )

    click.echo("Done! Check pee_log.csv and squat_log.csv for results.")


if __name__ == "__main__":
    cli()
