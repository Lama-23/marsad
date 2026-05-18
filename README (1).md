# Smart Road Inspection System — MARASD

> AI-powered road defect detection, severity scoring, and automated maintenance responsibility assignment — built for Saudi Arabian smart city infrastructure.

---

## Overview

MARASD combines computer vision, GPS telemetry, and GIS spatial analysis into a single inspection pipeline. It detects road damage from dashcam video, estimates defect severity, and automatically determines whether a contractor or municipality is responsible for repairs.

---

## System Pipeline

```
Video Input
    ↓
Frame Extraction
    ↓
Defect Detection  ──────→  YOLOv8m (4 classes)
    ↓
Depth Estimation  ──────→  MiDaS (potholes)
    ↓
GPS Mapping
    ↓
Risk Scoring
    ↓
GIS Geofencing
    ↓
Responsibility Assignment
```

---

## Detected Defects

| Class | Code | Description |
|---|---|---|
| Longitudinal Crack | D00 | Parallel to road direction |
| Transverse Crack | D10 | Perpendicular to road direction |
| Alligator Crack | D20 | Interconnected crack network |
| Pothole | D40 | Surface depression / material loss |

---

## Responsibility Logic

- Defect within **active licensed project boundary** → Contractor is responsible
- Otherwise → **Municipality** is responsible

Supported entities: SEC · NWC · TSPs · GASCO

---

## Detection Model — YOLOv8m

![Inference Sample](assets/inference_sample.png)

| | Baseline | This Model |
|---|---|---|
| Architecture | YOLOv8s | **YOLOv8m** |
| Training epochs | 50 | **120** |
| Augmentation | Default | **Custom (Saudi conditions)** |
| Expected mAP@0.5 | ~0.63 | **~0.70 – 0.75+** |

**25.8M params · 79.1 GFLOPs · 640×640**

### Training Strategy

Two-phase fine-tuning on [RDD2022](https://github.com/sekilab/RoadDamageDetector) (37,230 train / 3,286 val images):

**Phase 1** — Backbone frozen · 15 epochs · lr = 1e-3  
**Phase 2** — Full fine-tuning · 105 epochs · lr = 2e-4 · early stopping (patience = 20)

Optimizer: AdamW · weight decay = 5e-4 · batch = 16

### Augmentation

Custom Albumentations pipeline simulating Saudi road conditions (applied to 80% of images):

| Condition | Transforms |
|---|---|
| Harsh sunlight | `RandomBrightnessContrast` (brighten only) + `RandomGamma` |
| Dust & sand | `GaussNoise` + `ISONoise` + `HueSaturationValue` + `RGBShift` |
| Heat haze | `ElasticTransform` (alpha=30, subtle) |
| Motion blur | `MotionBlur` / `GaussianBlur` |
| Occlusion | `CoarseDropout` |
| Geometry | Flip · Rotate ±10° · Perspective · ShiftScaleRotate |

![Augmentation Samples](assets/augmentation_preview.png)

### Results

![Training Curves](assets/training_curves.png)
![Class Distribution](assets/class_distribution.png)

---

## Tech Stack

`Python` · `YOLOv8` · `MiDaS` · `OpenCV` · `Albumentations` · `ONNX Runtime` · `GIS Simulation`

---

## Future Work

- Real-time GIS and IoT integration
- Heatmap dashboard for defect analytics
- Object tracking to eliminate duplicate detections
- Edge deployment on Raspberry Pi

---

## Limitations

- No night-time training data
- Pothole class is underrepresented in RDD2022
- MiDaS depth is relative, not metric
- GIS boundaries are simulated, not live municipal data
