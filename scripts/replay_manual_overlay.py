"""Replay the manual webcam script over the first 120 frames of the recorded demo."""
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
    samples = []
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
        if frame is None or frame.size == 0:
            raise RuntimeError("The overlay script emitted an empty frame")
        samples.append(1000 * (time.perf_counter() - started))

    try:
        with (
            patch.object(cv2, "VideoCapture", return_value=ReplayCapture()),
            patch.object(cv2, "waitKey", return_value=-1),
            patch.object(cv2, "imshow", side_effect=collect_frame),
            patch.object(cv2, "destroyAllWindows"),
            patch.object(Image, "open", wraps=Image.open) as opened,
        ):
            runpy.run_path(str(ROOT / "AROVERLAY_FinalSP_WithMetrices-RealTime.py"), run_name="__main__")
    finally:
        capture.release()

    overlay_frames = sum(str(call.args[0]).startswith("hairstyles/") for call in opened.call_args_list)
    if len(samples) != FRAME_LIMIT or not overlay_frames:
        raise RuntimeError(f"Replay emitted {len(samples)} frames and opened {overlay_frames} overlays")
    print(json.dumps({
        "clip": VIDEO.name, "script": "manual", "python": platform.python_version(),
        "frames": len(samples), "frames_loading_overlay": overlay_frames,
        "median_frame_ms": round(statistics.median(samples), 2),
        "p95_frame_ms": round(sorted(samples)[math.ceil(.95 * len(samples)) - 1], 2),
    }))


if __name__ == "__main__":
    main()
