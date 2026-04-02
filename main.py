import asyncio
import base64
import logging
import queue
import threading
import time

import click
import cv2
from nicegui import app, ui

# Import your class from your other file
from camera_service import Camera


# 1. THE PRODUCER (Thread-based)
def camera_thread_worker(
    source, resolution, frame_queue: queue.Queue, stop_event: threading.Event
):
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
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
            else:
                time.sleep(0.1)  # Wait if camera is struggling

            # Tiny sleep to let other threads work
            time.sleep(0.01)
    finally:
        print("Releasing camera...")
        cam.release()


# 2. THE UI LOGIC
def start_ui(source, res, port, frame_queue: queue.Queue, stop_event: threading.Event):
    @ui.page("/")
    def index():
        ui.label(f"Streaming: {source} ({res})").classes("text-h4")
        placeholder = ui.interactive_image().classes("w-full max-w-2xl border-2")

        async def update_stream():
            while True:
                # Non-blocking check of the queue
                try:
                    frame = frame_queue.get_nowait()
                    # Convert to JPEG for the browser
                    _, buffer = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                    placeholder.set_source(f"data:image/jpeg;base64,{b64}")
                except queue.Empty:
                    pass
                # Update at roughly 30 FPS
                await asyncio.sleep(0.03)

        ui.timer(0.1, update_stream, once=True)

    # Cleanup when NiceGUI closes
    app.on_shutdown(lambda: stop_event.set())

    ui.run(port=port, title="Pi 5 Threaded Stream", reload=False)


# 3. THE CLI ENTRY POINT
@click.command()
@click.option("--source", default="usb0", help="usb0 or picamera0")
@click.option("--res", default="640x480", help="Resolution (WxH)")
@click.option("--port", default=8000, help="Web port")
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
        args=(source, res, frame_queue, stop_event),
        daemon=True,  # Thread dies if the main script stops
    )
    camera_thread.start()

    # Start the NiceGUI loop
    start_ui(source, res, port, frame_queue, stop_event)


if __name__ in {"__main__", "__mp_main__"}:
    main()
