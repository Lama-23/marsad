import cv2
import numpy as np
import onnxruntime as ort

class Detection:
    """Represents a single road defect detection with its properties."""
    def __init__(self, cls_id, cls_name, conf, bbox, risk_score, risk_level):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.conf = conf
        self.bbox = bbox
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.thumb_path = ""

class AIEngine:
    """Handles AI Inference for YOLO (Object Detection) and MiDaS (Depth Estimation)."""
    
    def __init__(self, yolo_path, midas_path):
        # Initialize Inference Sessions using ONNX Runtime
        # This approach significantly boosts performance on CPU-based devices like Raspberry Pi 5
        self.yolo_session = ort.InferenceSession(yolo_path)
        self.midas_session = ort.InferenceSession(midas_path)

        # Import YOLO here to avoid redundant re-imports during frame processing
        from ultralytics import YOLO
        # Load the model once during initialization for maximum efficiency
        self.yolo_model = YOLO(yolo_path)

        # Risk weights for different defect categories
        # 0: LongitudinalCrack, 1: TransverseCrack, 2: AlligatorCrack, 3: Pothole
        self.risk_weights = {0: 0.5, 1: 0.6, 2: 0.75, 3: 1.0}
        self.class_names = {0: 'LongitudinalCrack', 1: 'TransverseCrack', 2: 'AlligatorCrack', 3: 'Pothole', 4: 'Repair'}

    def process_frame(self, frame, frame_index):
        """Processes a single video frame to detect defects and estimate depth/risk."""
        
        # 1. YOLO Inference
        # We use the pre-loaded model for speed
        results = self.yolo_model(frame, verbose=False)[0]

        # 2. Depth Estimation via ONNX
        # This provides a relative depth map for better damage assessment
        depth_map = self._get_depth_onnx(frame)

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)

                # Extract depth data for the specific detected region (Bounding Box)
                crop = depth_map[max(0, y1):y2, max(0, x1):x2]
                avg_depth = np.mean(crop) if crop.size > 0 else 0

                # Perform risk mathematical calculations
                risk_score, risk_level = self._calculate_risk(
                    cls_id, conf, bbox, frame.shape, avg_depth
                )

                # Create the detection object used by the Visualizer
                detections.append(Detection(
                    cls_id=cls_id,
                    cls_name=self.class_names.get(cls_id, "Unknown"),
                    conf=conf,
                    bbox=bbox,
                    risk_score=risk_score,
                    risk_level=risk_level
                ))

        return detections

    def _get_depth_onnx(self, frame):
        """
        Generates a depth map using ONNX Runtime.
        Typically 3x faster than PyTorch on standard CPU hardware.
        """
        # Preprocessing: Convert to RGB, Resize, and Normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, :]

        # Run ONNX Inference
        onnx_inputs = {self.midas_session.get_inputs()[0].name: img_input}
        depth = self.midas_session.run(None, onnx_inputs)[0]

        # Resize back to original frame dimensions
        depth_resized = cv2.resize(depth.squeeze(), (frame.shape[1], frame.shape[0]))
        return depth_resized

    def _calculate_risk(self, cls_id, conf, bbox, img_shape, avg_depth):
        """Calculates a normalized risk score (0-100) based on multiple parameters."""
        x1, y1, x2, y2 = bbox
        img_h, img_w = img_shape[:2]
        
        # Calculate area ratio of the defect relative to the frame size
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

        # Weighted calculation logic
        type_score = self.risk_weights.get(cls_id, 0.5) * 30
        size_score = min(30, np.log1p(area_ratio * 200) * 10)
        conf_score = conf * 10
        
        # Add depth score specifically for Potholes (Class ID 3)
        depth_score = min(30, (avg_depth / 250) * 30) if cls_id == 3 else 0

        # Clip total to 0-100 range
        total = np.clip(type_score + size_score + conf_score + depth_score, 0, 100)

        # Determine qualitative risk level
        level = 'safe' if total < 35 else 'moderate' if total < 70 else 'danger'
        
        return round(float(total), 1), level