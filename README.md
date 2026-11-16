# Real-time AR hairstyle overlay

This computer vision project fits a hairstyle to a moving face in a webcam feed. MediaPipe Face Mesh locates the forehead and temples. Their separation sets the overlay width, and the angle between the temples rotates the hairstyle as the head tilts.

**[Watch the 11-style demo](FinalVideo-11Styles_Metrices.mp4)** · [View the project slides](Final-AR_smallProject.pptx)

## Explore the implementation

| File | What to look for |
| --- | --- |
| [Manual overlay](AROVERLAY_FinalSP_WithMetrices-RealTime.py) | Follow a frame from OpenCV capture through landmark detection, resizing, rotation and alpha blending. |
| [Automatic style selection](AROVERLAY_AllMetrices-RealTime.py) | Compares the candidate overlay boxes with a forehead box. It can switch styles when the best overlap exceeds 0.35. |
| [Four-image prototype](AR_overlay_4images-RealTime.py) | Earlier version of the webcam overlay. |

## Reproduce the webcam demo

Use Python 3.11 and run these commands from the repository root. The preparation script creates eleven transparent alignment markers at the paths used by both 11-style webcam scripts. It preserves any hairstyle PNGs you already have. The recorded demo uses the original hairstyle artwork. The supplied generator draws numbered arcs and centre marks so you can check tilt, scale and style selection with a fresh checkout.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python scripts/create_sample_overlays.py
python AROVERLAY_FinalSP_WithMetrices-RealTime.py
```

The interactive demo needs a display and a webcam at device index 0. On Windows, activate the environment with `.venv\\Scripts\\activate`. To replay the manual overlay without a camera or window, run `python scripts/replay_manual_overlay.py` after creating the markers.

The replay feeds the first 120 frames of the existing demonstration video through the manual script and records median and p95 processing time. Timing starts at frame decode and ends when the script emits the frame. It excludes webcam exposure time and screen display. The source video already contains rendered overlays, so this replay measures execution on that clip. Independent reference landmarks are still needed to measure placement error.

On 16 November 2026 the Ubuntu 22.04 CI runner (AMD EPYC 9V45, Python 3.11.18) processed 120 frames and loaded an overlay in every frame. Median frame time was **5.28 ms** and p95 was **6.20 ms** in this single headless replay. [Recorded CI run](https://github.com/Jeeson-Kurian/AR-Codebase/actions/runs/43970691002)

## Read the on-screen measurements

FPS measures the time between successive loop iterations on the machine running the demo. IoU measures geometric overlap between the constructed forehead and hairstyle boxes. The automatic variant's “confidence” value is derived from the number of landmarks returned. Assessing placement accuracy would require labelled frames and a separate evaluation.
