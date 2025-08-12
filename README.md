# AgriSlide — Robust Crop Row Extraction Beyond Classical Vision
**ROS 2 • Gazebo • Python • Motion Planning**

AgriSlide compares two ways to extract crop rows from **machine-learning vegetation masks** and convert them into navigation errors for **GPS-free row following**:

- **AgriSlide (Sliding Windows on ML Mask)** — stable, fast row tracking by recentering windows up the mask  
- **Classical (Canny + Hough on ML Mask)** — lightweight line voting and averaging

**Published topics (controller-friendly):**
- `/nav/lat_error` – sideways offset (pixels; +right, −left)  
- `/nav/ang_error` – heading misalignment (degrees; +right tilt, −left tilt)

> **Media note:** this repo will include **screenshots** (PNGs/GIFs) rather than videos.



## Features
- **DeepLabv3+** mask → **rows** → **nav line** → **errors**
- **Sliding-window** tracker robust to weeds, gaps, and lighting
- **Classical Canny + Hough** comparison on the same masks
- **ROS 2** publishers for lateral/heading errors
- Designed for **real-time** use in the field or simulation

---

## Quickstart

- **Camera** — publish a color image on `/camera/color/image_raw` (from Gazebo or a real camera).
- **Run AgriSlide pipeline** — use `src/ml_classical_pipeline.py` in your ROS 2 node (or `lane_detection_node.py`).
  - DeepLabv3+ → mask  
  - Sliding windows → polylines & midline  
  - Compute errors → publish topics
- **(Optional) Compare with Classical** — run `src/inference_lane_detector.py` to compute the midline via Canny + Hough and publish errors.

---

## Configuration

### AgriSlide (sliding windows)
- `num_windows` (e.g., 10)  
- `margin` (px half-width, e.g., 20)  
- `minpix` (min points to recenter, e.g., 70)  
- Polynomial fit order (quadratic used here)

### Classical (Canny + Hough)
- Canny thresholds (`low`, `high`)  
- Hough (`minLineLength`, `maxLineGap`, `threshold`)  
- Angle threshold to ignore near-horizontal noise  
- Offsets \((\delta_t, \delta_b)\) for single-side case

### Common
- Processing size (e.g., `512×512`)  
- ROI cropping box  
- EMA smoothing `alpha` (e.g., `0.1`)

---

## Performance

| Metric                   | AgriSlide (Sliding) | Canny + Hough |
|--------------------------|---------------------|---------------|
| Runtime @ 512×512 (FPS) | **15**              | **11**        |

> On our setup, AgriSlide is faster by avoiding per-frame global Hough voting and relying on cheap, stable window recentering.

---

## Screenshots (to be added)

> **Tip:** GitHub doesn’t allow YouTube to play inline in README, so we use a clickable thumbnail.

**▶️ Watch the demo (YouTube)**  
[![AgriSlide Demo](https://img.youtube.com/vi/yABlitEfaKY/hqdefault.jpg)](https://www.youtube.com/watch?v=yABlitEfaKY)

- AgriSlide overlay (mask + polylines + midline)  
  ![AgriSlide Overlay](assets/screenshots/agrislide_overlay.png)

- Classical overlay (edges + lines + midline)  
  ![Classical Overlay](assets/screenshots/classical_overlay.png)

- Error traces (lateral & angular vs time)  
  ![Error Traces](assets/screenshots/error_traces.png)

---

## Troubleshooting

- **No mask / all black:** confirm `models/deeplab_full_model.pt` path and device (CPU/GPU) match your install.  
- **Jittery errors:** raise `minpix`, increase `margin`, or increase EMA `alpha` slightly.  
- **No Hough lines:** lower `hough_threshold` or `canny_high_threshold`; ensure mask isn’t over-eroded.
