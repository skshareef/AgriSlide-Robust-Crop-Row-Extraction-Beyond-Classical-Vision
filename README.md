# AgriSlide — Robust Crop Row Extraction Beyond Classical Vision
**ROS 2 • Gazebo • Python • Motion Planning**

AgriSlide compares two ways to extract crop rows from **machine-learning vegetation masks** and convert them into navigation errors for **GPS-free row following**:

- **AgriSlide (Sliding Windows on ML Mask)** — stable, fast row tracking by recentering windows up the mask  
- **Classical (Canny + Hough on ML Mask)** — lightweight line voting and averaging

**Published topics (controller-friendly):**
- `/nav/lat_error` – sideways offset (pixels; +right, −left)  
- `/nav/ang_error` – heading misalignment (degrees; +right tilt, −left tilt)

> **Media note:** this repo will include **screenshots** (PNGs/GIFs) rather than videos.

---

## Table of Contents
- [Features](#features)
- [Repo Structure](#repo-structure)
- [Methods & Intuition](#methods--intuition)
- [Navigation Errors (Clear Formulas)](#navigation-errors-clear-formulas)
- [ROS 2 Interface](#ros-2-interface)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Performance](#performance)
- [Screenshots (to be added)](#screenshots-to-be-added)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citations & Acknowledgments](#citations--acknowledgments)

---

## Features
- **DeepLabv3+** mask → **rows** → **nav line** → **errors**
- **Sliding-window** tracker robust to weeds, gaps, and lighting
- **Classical Canny + Hough** comparison on the same masks
- **ROS 2** publishers for lateral/heading errors
- Designed for **real-time** use in the field or simulation

---

## Repo Structure
```text
agrislide/
├─ models/
│  └─ deeplab_full_model.pt         # DeepLabv3+ weights (place here)
├─ src/
│  ├─ ml_classical_pipeline.py      # AgriSlide: ML mask + sliding windows + errors
│  ├─ inference_lane_detector.py    # Classical: ML mask + Canny + Hough + errors
│  ├─ Deeplabv3Plus.py              # Model definition (if needed for state_dict)
│  ├─ lane_detection_node.py        # (Optional) ROS 2 node wiring camera → topics
├─ launch/
│  └─ agrislide.launch.py           # (Optional) ROS 2 launch
├─ assets/
│  └─ screenshots/                  # Add PNG/GIF screenshots here
├─ requirements.txt
├─ package.xml / setup.py (if ROS 2 pkg)
└─ README.md

```


Quickstart
Camera — publish a color image on /camera/color/image_raw (from Gazebo or a real camera).

Run AgriSlide pipeline — use src/ml_classical_pipeline.py in your ROS 2 node (or lane_detection_node.py).

DeepLabv3+ → mask

Sliding windows → polylines & midline

Compute errors → publish topics

(Optional) Compare with Classical — run src/inference_lane_detector.py to compute the midline via Canny + Hough and publish errors.

Configuration
AgriSlide (sliding windows)

num_windows (e.g., 10)

margin (px half-width, e.g., 20)

minpix (min points to recenter, e.g., 70)

Polynomial fit order (quadratic used here)

Classical (Canny + Hough)

Canny thresholds (low, high)

Hough (minLineLength, maxLineGap, threshold)

Angle threshold to ignore near-horizontal noise

Offsets 
(
𝛿
𝑡
,
𝛿
𝑏
)
(δ 
t
​
 ,δ 
b
​
 ) for single-side case

Common

Processing size (e.g., 512×512)

ROI cropping box

EMA smoothing alpha (e.g., 0.1)

Performance
Metric	AgriSlide (Sliding)	Canny + Hough
Runtime @ 512×512 (FPS)	15	11

On our setup, AgriSlide is faster by avoiding per-frame global Hough voting and relying on cheap, stable window recentering.

Screenshots (to be added)
Place images under assets/screenshots/ and reference them here:

AgriSlide overlay (mask + polylines + midline)
![AgriSlide Overlay](assets/screenshots/agrislide_overlay.png)

Classical overlay (edges + lines + midline)
![Classical Overlay](assets/screenshots/classical_overlay.png)

Error traces (lateral & angular vs time)
![Error Traces](assets/screenshots/error_traces.png)

Troubleshooting
No mask / all black: confirm models/deeplab_full_model.pt path and device (CPU/GPU) match your install.

Jittery errors: raise minpix, increase margin, or increase EMA alpha slightly.

No Hough lines: lower hough_threshold or canny_high_threshold; ensure mask isn’t over-eroded.

