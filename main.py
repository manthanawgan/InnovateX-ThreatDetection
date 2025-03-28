import cv2
import numpy as np
from ultralytics import YOLO

def detect_threats():
    cap = cv2.VideoCapture(0)

    model = YOLO('yolov8n.pt')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        results = model(frame)
        
        annotated_frame = results[0].plot()
        
        cv2.imshow('Threat Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_threats()