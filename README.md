# 🪟 Window Detection & Autonomous Flight with Tello Drone

> An autonomous drone system using **YOLO segmentation** and **PID control** to detect a window, align with it, approach it, and pass through it — running on a **DJI Tello** drone.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [State Machine](#-state-machine)
- [Keyboard Controls](#-keyboard-controls)
- [Safety Notes](#-safety-notes)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project connects to a **DJI Tello** drone, streams live video, and runs a custom-trained YOLO model (`best.pt`) to detect a window. Using **three PID controllers** (Yaw, Roll, Throttle) plus a forward approach phase, the drone:

1. **Searches** for the window (360° rotation + vertical scan)
2. **Aligns** the window with the camera center
3. **Approaches** the window until it fills a threshold of the frame
4. **Passes** through the window (forward burst)
5. **Lands** or completes the mission

Designed for **CPU inference** with performance tuning knobs for FPS vs. accuracy.

---

## ✨ Features

- ✅ **YOLO segmentation** with mask polygon extraction
- ✅ **Geometric yaw estimation** — uses left vs. right edge length of the window quad
- ✅ **Simultaneous YAW + ROLL alignment**
- ✅ **Three independent PID controllers** with windup limits and output smoothing
- ✅ **Search mode** — 360° rotation + vertical scan
- ✅ **Approach phase** — forward motion with continuous centering
- ✅ **Pass phase** — detects when the drone has passed through the window
- ✅ **Live HUD overlay** — errors, commands, stage, FPS
- ✅ **Emergency stop** (`E` key) and reset (`R` key)
- ✅ **Battery safety check** before takeoff

---

## 🎬 Demo

> Add a GIF or video here showing the drone passing through a window.

```
docs/demo.gif
```

---

## 🛠 Hardware Requirements

| Component | Notes |
|-----------|-------|
| **DJI Tello** drone | Battery ≥ 20% |
| **Wi-Fi adapter** | To connect to Tello's AP |
| **PC / Laptop** | Python 3.8+ |

> ⚠️ Connect the PC to Tello's Wi-Fi **before** running.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/window-tello.git
cd window-tello
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the trained model

Place your YOLO **segmentation** model `best.pt` in the project root.

> 💡 Model files (>100 MB) should be uploaded via **Git LFS** or attached as a **GitHub Release** asset.

### 4. Run

```bash
python window_tello.py
```

---

## ⚙️ Configuration

Key parameters at the top of `window_tello.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | YOLO segmentation model path |
| `CONFIDENCE` | `0.5` | Minimum detection confidence |
| `INFERENCE_SIZE` | `640` | YOLO input size (320 = faster) |
| `FLIGHT_MODE` | `True` | Set to `False` for dry-run (no RC commands) |
| `MIN_BATTERY` | `20` | Min battery % to allow takeoff |
| `SEARCH_360_TIME` | `15.0` | Seconds for one full rotation |
| `YAW_THRESHOLD` | `5.0` | % edge-length difference to consider yaw centered |
| `ROLL_INVERT` | `True` | Flip roll direction if needed |
| `YAW_GEOMETRY_INVERT` | `False` | Flip yaw direction if correction is reversed |
| `AREA_RATIO_THRESHOLD` | `0.35` | Stop approach when window fills this fraction |
| `PASS_FORWARD_SPEED` | `40` | Forward burst speed to pass through |

### PID Gains

| Axis | Kp | Ki | Kd | Max Speed |
|------|----|----|----|-----------|
| Yaw | 0.8 | 0.0 | 0.0 | 35 |
| Roll | 0.1 | 0.0 | 0.0 | 20 |
| Throttle | 0.4 | 0.05 | 0.0 | 25 |

---

## 🧠 How It Works

### Detection Pipeline

```
Tello frame → resize 640×480 → YOLO predict
    → pick largest area box
    → extract mask polygon
    → compute window center
    → extract quad corners from mask
    → feed to Aligner.update()
```

### Error Computation

| Error | Formula |
|-------|---------|
| Yaw (raw) | `window_cx − camera_cx` |
| Yaw (geometric) | `100 × (right_len − left_len) / avg_len` |
| Throttle | `window_cy − camera_cy` |
| Roll | `(dist_right − dist_left) / total` |

### Smoothing

All errors pass through an **EMA filter**:
```python
smoothed = (1 − α) × smoothed + α × raw    # α = 0.3
```

---

## 🔄 State Machine

```
START
  │
  ▼
SEARCH 360 ─── found ──▶ YAW + ROLL ──▶ THROTTLE ──▶ APPROACH ──▶ PASS ──▶ DONE
  │                          ▲
  │ 360 done                 │
  ▼                          │
SEARCH UP ──────────────────┘
```

| Stage | Behavior |
|-------|----------|
| **SEARCH 360** | Rotates to find window |
| **SEARCH UP** | Climbs, then restarts 360° |
| **YAW+ROLL** | Simultaneous yaw (geometric) + roll alignment |
| **THROTTLE** | Vertical centering |
| **APPROACH** | Forward motion with continuous centering |
| **PASS** | Forward burst through the window |
| **DONE** | Stops all motion |

---

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit (lands drone) |
| `R` | Reset aligner state |
| `E` | **Emergency stop** |

---

## 🛡 Safety Notes

1. **Always test with `FLIGHT_MODE = False` first.**
2. **Battery check** prevents takeoff below 20%.
3. **Emergency stop** — press `E` to abort; `finally` block lands the drone.
4. **Indoor use only** — Tello has no GPS.
5. **Prop guards recommended.**
6. **Wi-Fi interference** — keep a clear line of sight.
7. **Never run unsupervised** — PIDs can oscillate.

> **The author assumes no liability for damage or injury.**

---

## 🐛 Troubleshooting

### Window not detected

| Symptom | Fix |
|---------|-----|
| No boxes | Check `model.names`; confirm model trained on window |
| Low confidence | Lower `CONFIDENCE` to `0.25` for testing |
| Blurry frames | Hover stable, improve lighting |
| No masks | Use a `-seg` model, not detection-only |

**Debug tip:**
```python
if results and results[0].boxes is not None:
    print(f"detections: {len(results[0].boxes)}")
```

### Drone moves wrong direction

- Flip `ROLL_INVERT` or `YAW_GEOMETRY_INVERT`

### Yaw never centers

- Your model likely isn't segmentation, so `window_corners` is `None`. Either use a `-seg` model or switch yaw source to `yaw_err_raw`.

### Low FPS

- Reduce `INFERENCE_SIZE` to `320`
- Enable CUDA if available

### `index.lock` error in Git

```bash
rm -f .git/index.lock
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [djitellopy](https://github.com/damiafuentes/DJITelloPy)
- [OpenCV](https://opencv.org/)

---

⭐ **If this project helped you, please star the repo!**
