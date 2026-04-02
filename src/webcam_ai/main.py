import logging
import queue
import threading
import time

import click
import cv2
from fastapi import Response
from nicegui import app, ui

# Import your class from your other file
from webcam_ai.camera_service import Camera, StreamState

state = StreamState()


# 1. THE PRODUCER (Thread-based)
def camera_thread_worker(source, resolution, stop_event: threading.Event):
    """
    Runs in a background thread.
    Continuously captures frames and puts them in the queue.
    """
    logging.info(f"Starting camera thread: {source} at {resolution}")
    cam = Camera(source=source, resolution=resolution)

    try:
        while not stop_event.is_set():
            ret, frame = cam.read()
            if ret:
                # Keep only the latest frame in the queue to prevent lag
                _, buffer = cv2.imencode(".jpg", frame)
                state.latest_jpeg = buffer.tobytes()
            else:
                time.sleep(0.1)  # Wait if camera is struggling

            # Tiny sleep to let other threads work
            time.sleep(0.01)
    finally:
        logging.info("Releasing camera...")
        cam.release()


@app.get("/video/stream")
def video_stream():
    if state.latest_jpeg is not None:
        return Response(content=state.latest_jpeg, media_type="image/jpeg")
    return Response(status_code=404)


# 2. THE UI LOGIC
def start_ui(source, res, port, stop_event: threading.Event):
    @ui.page("/")
    def index():
        ui.label(f"Streaming: {source} ({res})").classes("text-h4")
        video_image = ui.interactive_image("/video/stream").classes(
            "w-full max-w-2xl border-2"
        )

        ui.timer(0.1, callback=video_image.force_reload)

    # Cleanup when NiceGUI closes
    app.on_shutdown(lambda: stop_event.set())

    ui.run(port=port, title="Pi 5 Threaded Stream", reload=False)


# 3. THE CLI ENTRY POINT
@click.command()
@click.option("--source", default="usb0", help="usb0 or picamera0")
@click.option("--res", default="640x480", help="Resolution (WxH)")
@click.option("--port", default=8080, help="Web port")
def main(source, res, port):
    # Standard Python Queue (Thread-safe)
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Create and start the camera thread
    camera_thread = threading.Thread(
        target=camera_thread_worker,
        args=(source, res, stop_event),
        daemon=True,  # Thread dies if the main script stops
    )
    camera_thread.start()

    # Start the NiceGUI loop
    start_ui(source, res, port, stop_event)


if __name__ in {"__main__", "__mp_main__"}:
    main()
