# Smart Road Inspection System — MARASD

> AI-powered road defect detection, severity scoring, and automated maintenance responsibility assignment — built for Saudi Arabian smart city infrastructure.

---

## Overview

MARASD combines computer vision, GPS telemetry, and GIS spatial analysis into a single inspection pipeline. It detects road damage from dashcam video, estimates defect severity, and automatically determines whether a contractor or municipality is responsible for repairs.

---

## System Pipeline


<img width="825" height="398" alt="Screenshot 2026-05-18 at 11 44 52 AM" src="https://github.com/user-attachments/assets/7b62d6e6-c9ac-47a6-9b7d-8bb0284ed6c8" />


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


## Responsibility Logic

- Defect within **active licensed project boundary** → Contractor is responsible
- Otherwise → **Municipality** is responsible

---


## Detected Defects

| Class | Code | Description |
|---|---|---|
| Longitudinal Crack | D00 | Parallel to road direction |
| Transverse Crack | D10 | Perpendicular to road direction |
| Alligator Crack | D20 | Interconnected crack network |
| Pothole | D40 | Surface depression / material loss |

## Dataset Distribution & Split


<img width="1142" height="402" alt="1" src="https://github.com/user-attachments/assets/8228364f-728f-4250-8e04-930b0e61348c" />


---

## Detection Model — YOLOv8m



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

<img width="903" height="680" alt="2" src="https://github.com/user-attachments/assets/36a8ce3b-9697-469c-8503-1abd129636cb" />

### Results

<img width="890" height="761" alt="3" src="https://github.com/user-attachments/assets/14af4466-9ddd-47bc-a425-ab25d925f48c" />
<img width="826" height="247" alt="4" src="https://github.com/user-attachments/assets/d1bee61b-1fac-43e4-9175-6343303acbb0" />


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


  ---------
  ## note
  To download the fine-tuned model, visit the Kaggle Model repository:
  
  https://www.kaggle.com/models/lamaww/yolov8m-rdd-saudi-enhanced
