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
