from ultralytics import YOLO
import cv2
import numpy as np

class VisionDetector:
    def __init__(self, model_path='yolov8m.pt'):
        # Load a pretrained YOLOv8n model
        self.model = YOLO(model_path)
        # Classes we care about: 0=person, 43=knife, 76=scissors
        # Note: Standard COCO dataset doesn't have 'gun', but 'knife' is available.
        self.target_classes = [0, 43, 76] 
        print("VisionDetector initialized (v2.0 with draw_detections)") 

    def detect(self, frame):
        """
        Run inference on a frame.
        Returns:
            - annotated_frame: numpy array with boxes drawn
            - motion_score: float 0-1 based on detection confidence and count
            - detections: list of dicts
        """
        results = self.model(frame, verbose=False)
        result = results[0]
        
        detections = []
        max_conf = 0.0
        person_count = 0

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls in self.target_classes:
                person_count += 1
                max_conf = max(max_conf, conf)
                detections.append({
                    "class": result.names[cls],
                    "confidence": conf,
                    "box": box.xyxy[0].tolist()
                })

        # Threat Logic:
        # - Person ONLY: Score 0.0 (Safe)
        # - Weapon (Knife/Scissors): Score 1.0 (Danger)
        
        has_weapon = False
        for d in detections:
            if d['class'] in ['knife', 'scissors']:
                has_weapon = True
                break
        
        if has_weapon:
            motion_score = 1.0
        else:
            # If only people are seen, it's NOT a threat.
            motion_score = 0.0
        
        # Draw annotations on the frame
        annotated_frame = self.draw_detections(frame, detections)
        
        return annotated_frame, motion_score, detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on a frame manually.
        Useful for persisting detections across skipped frames.
        """
        annotated_frame = frame.copy()
        
        for d in detections:
            x1, y1, x2, y2 = map(int, d['box'])
            conf = d['confidence']
            cls_name = d['class']
            
            # Color based on class (Red for weapons, Green for person)
            color = (0, 0, 255) if cls_name in ['knife', 'scissors'] else (0, 255, 0)
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return annotated_frame
