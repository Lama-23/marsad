import cv2
import logging
import json
from pathlib import Path
from ai_engine import AIEngine
from gis_logic import GISLayer, GPSSimulator, ResponsibilityEngine
from visualizer import Visualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Main")

BASE_DIR = Path(__file__).resolve().parent

# --- Configuration ---
VIDEO = BASE_DIR / "data/sample_video.mp4"
YOLO_MODEL = BASE_DIR / "models/best.onnx"      
MIDAS_MODEL = BASE_DIR / "models/midas_small.onnx" 
DATA_PATH = BASE_DIR / "data/licenses.json"  
SHOW = True

def run_pipeline(video_path, yolo_path, midas_path):
    logger.info("--- Starting Smart Road Inspection System (MARASD) ---")

    # Initialize engines
    ai_engine = AIEngine(yolo_path=str(yolo_path), midas_path=str(midas_path))
    gis_layer = GISLayer(str(DATA_PATH))
    gps_sim = GPSSimulator(str(DATA_PATH))
    resp_engine = ResponsibilityEngine(gis_layer)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video file at: {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Output paths
    output_video = str(BASE_DIR / "data/final_output.mp4")
    output_html = str(BASE_DIR / "data/marsad_gis.html")
    output_json = str(BASE_DIR / "data/inspection_results.json")

    # Initialize Visualizer
    viz = Visualizer(output_path=output_video, width=width, height=height, fps=fps)

    # Data collection for GIS Dashboard
    all_detections_data = []

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video file reached.")
            break

        if frame_index % 30 == 0:
            logger.info(f"Processing frame {frame_index}...")

        # 1. Get current metadata
        gps_point = gps_sim.next()
        decision = resp_engine.decide(gps_point)
        
        # 2. Run AI Inference
        detections = ai_engine.process_frame(frame, frame_index)

        # 3. Process Visualization (Pass frame_index here to fix the error)
        # This will also fill d.thumb_path inside each detection
        annotated_frame = viz.process(frame, detections, gps_point, decision, frame_index)

        # 4. Log data for GIS Dashboard
        for d in detections:
            all_detections_data.append({
                "lat": gps_point['lat'],
                "lng": gps_point['lng'],
                "class": d.cls_name,
                "risk_score": d.risk_score,
                "risk_level": d.risk_level,
                "confidence": float(d.conf),
                "party": decision.party,
                "timestamp": gps_point['simulated_date'],
                "depth": getattr(d, 'depth_cm', 15.5),
                "thumb_path": d.thumb_path    # مضاف لعرض الصور
            })

        if SHOW:
            cv2.imshow("MARASD - Road Inspection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    # Cleanup and Save Results
    cap.release()
    viz.release()
    cv2.destroyAllWindows()

    # Save JSON log for future analysis
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_detections_data, f, ensure_ascii=False, indent=4)

    # Generate the interactive HTML Dashboard with the gathered data
    viz.generate_html_dashboard(all_detections_data, output_html)

    logger.info(f"Data saved to: {output_json}")
    logger.info(f"Dashboard generated at: {output_html}")
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    run_pipeline(VIDEO, YOLO_MODEL, MIDAS_MODEL)