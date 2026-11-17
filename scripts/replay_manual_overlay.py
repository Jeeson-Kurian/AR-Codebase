"""Replay a webcam script over the first 120 frames of the recorded demo."""
import argparse
import json
import math
import platform
import runpy
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "FinalVideo-11Styles_Metrices.mp4"
FRAME_LIMIT = 120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("manual", "automatic"), default="manual")
    mode = parser.parse_args().mode
    script = {
        "manual": "AROVERLAY_FinalSP_WithMetrices-RealTime.py",
        "automatic": "AROVERLAY_AllMetrices-RealTime.py",
    }[mode]
    samples = []
    loaded_in_frame = False
    overlay_frames = 0
    open_image = Image.open

    def track_image(path, *args, **kwargs):
        nonlocal loaded_in_frame
        image = open_image(path, *args, **kwargs)
        if isinstance(path, (str, Path)) and Path(path).parent == Path("hairstyles"):
            loaded_in_frame = True
        return image

    started = 0.0
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to decode {VIDEO.name}")

    class ReplayCapture:
        def isOpened(self):
            return len(samples) < FRAME_LIMIT and capture.isOpened()

        def read(self):
            nonlocal started
            started = time.perf_counter()
            return capture.read()

        def release(self):
            capture.release()

    def collect_frame(_window, frame):
        nonlocal overlay_frames, loaded_in_frame
        if frame is None or frame.size == 0:
            raise RuntimeError("The overlay script emitted an empty frame")
        samples.append(1000 * (time.perf_counter() - started))
        overlay_frames += int(loaded_in_frame)
        loaded_in_frame = False

    try:
        with (
            patch.object(cv2, "VideoCapture", return_value=ReplayCapture()),
            patch.object(cv2, "waitKey", return_value=-1),
            patch.object(cv2, "imshow", side_effect=collect_frame),
            patch.object(cv2, "destroyAllWindows"),
            patch.object(Image, "open", new=track_image),
        ):
            runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        capture.release()

    if len(samples) != FRAME_LIMIT or not overlay_frames:
        raise RuntimeError(f"Replay emitted {len(samples)} frames and opened {overlay_frames} overlays")
    print(json.dumps({
        "clip": VIDEO.name, "script": mode, "python": platform.python_version(),
        "frames": len(samples), "frames_loading_overlay": overlay_frames,
        "median_frame_ms": round(statistics.median(samples), 2),
        "p95_frame_ms": round(sorted(samples)[math.ceil(.95 * len(samples)) - 1], 2),
    }))


if __name__ == "__main__":
    main()
