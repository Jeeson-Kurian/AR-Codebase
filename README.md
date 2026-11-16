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

The 11-style scripts expect a webcam at device index 0, a display, OpenCV, Pillow, NumPy and a MediaPipe installation that exposes `mp.solutions.face_mesh`. They read eleven transparent PNGs from `hairstyles/hairstyle1.png` through `hairstyles/hairstyle11.png` relative to the working directory. The repository omits those assets and a pinned environment. To try the flow with your own images, supply all eleven at those paths and run the manual overlay from the repository root.

```sh
python AROVERLAY_FinalSP_WithMetrices-RealTime.py
```

## Read the on-screen measurements

FPS measures the time between successive loop iterations on the machine running the demo. IoU measures geometric overlap between the constructed forehead and hairstyle boxes. The automatic variant's “confidence” value is derived from the number of landmarks returned. Assessing placement accuracy would require labelled frames and a separate evaluation.
