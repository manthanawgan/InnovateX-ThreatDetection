import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import pygame
import os

class ThreatDetectionApp:
    def __init__(self):
        st.set_page_config(
            page_title="Threat Detection System", 
            page_icon=":detective:", 
            layout="wide"
        )
        
        if 'detection_active' not in st.session_state:
            st.session_state.detection_active = False

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        self.load_models()
        self.prepare_warning_sound()

    def prepare_warning_sound(self):
        """Prepare warning sound file"""
        try:
            self.warning_sound_path = tempfile.mktemp(suffix='.wav')

            from scipy.io import wavfile
            
            sample_rate = 44100
            duration = 1  # sec
            frequency = 800  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            beep = 0.5 * np.sin(2 * np.pi * frequency * t) * (np.sin(2 * np.pi * 4 * t) > 0)
            beep = (beep * 32767).astype(np.int16)   # Normalize to 16-bit range
            
            wavfile.write(self.warning_sound_path, sample_rate, beep)
        except Exception as e:
            st.warning(f"Could not prepare warning sound: {e}")
            self.warning_sound_path = None

    def load_models(self):
        """Load pre-trained and custom models"""
        try:
            self.default_model = YOLO('best.pt')          # Default YOLO model
            
            # Optional: Load custom trained model if exists
            try:
                self.custom_model = YOLO('my_model.pt')
            except:
                self.custom_model = None
                st.warning("⚠️ Custom model not found! Using default model.")
        except Exception as e:
            st.error(f"Error loading models: {e}")

    def render_sidebar(self):
        """Create sidebar for app controls"""
        st.sidebar.title("🕵️ Threat Detection Settings")

        detection_mode = st.sidebar.selectbox(
            "Detection Mode", 
            ["Thief Detection", "Weapon Detection", "Custom Detection"]
        )

        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5
        )

        camera_source = st.sidebar.selectbox(
            "Camera Source", 
            ["Default Camera", "IP Camera", "Upload Video"]
        )

        warning_type = st.sidebar.multiselect(
            "Warning Types",
            ["Sound Alert", "Visual Alert", "Popup Notification"],
            default=["Sound Alert", "Visual Alert"]
        )
        
        return {
            'mode': detection_mode,
            'confidence': confidence_threshold,
            'source': camera_source,
            'warnings': warning_type
        }

    def detect_threats(self, frame, model, confidence, mode):
        """Perform threat detection on a single frame"""
        results = model(frame, conf=confidence)[0]  # Run inference
        
        detected_classes = []
        if len(results.boxes) > 0:
            class_names = results.names
            for box in results.boxes:
                detected_classes.append(class_names[int(box.cls[0])])

        # Define the detection criteria
        if mode == "Thief Detection":
            threat_detected = any(cls in ["thief", "robber"] for cls in detected_classes)
        elif mode == "Weapon Detection":
            threat_detected = any(cls in ["gun", "knife"] for cls in detected_classes)
        else:  # Custom detection mode
            threat_detected = len(detected_classes) > 0

        annotated_frame = results.plot()  # Annotate frame
        return annotated_frame, results, threat_detected, detected_classes

    def trigger_warnings(self, settings, detected_classes):
        """Trigger various warning mechanisms"""
        detected_str = ', '.join(set(detected_classes)) if detected_classes else "Potential Threat"

        # Sound Alert
        if 'Sound Alert' in settings['warnings'] and self.warning_sound_path:
            try:
                warning_sound = pygame.mixer.Sound(self.warning_sound_path)
                warning_sound.set_volume(1.0)
                warning_sound.play()
            except Exception as e:
                st.warning(f"Could not play warning sound: {e}")

        # Visual Alert
        if 'Visual Alert' in settings['warnings']:
            warning_placeholder = st.empty()
            warning_placeholder.error(f"🚨 THREAT DETECTED: {detected_str}")
            time.sleep(2)
            warning_placeholder.empty()

        # Popup Notification
        if 'Popup Notification' in settings['warnings']:
            st.toast(f"⚠️ Threat Detected: {detected_str}")

    def run_camera_detection(self, settings):
        """Main camera detection logic"""
        st.title("🚨 Live Threat Detection")
        
        model = self.custom_model if settings['mode'] == "Custom Detection" and self.custom_model else self.default_model

        cap = cv2.VideoCapture(0)
        
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Detection")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Perform detection
            annotated_frame, results, threat_detected, detected_classes = self.detect_threats(
                frame, model, settings['confidence'], settings['mode']
            )
            
            # Trigger warnings if threat detected
            if threat_detected:
                self.trigger_warnings(settings, detected_classes)
            
            # Display processed frame
            if annotated_frame is not None:
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("Detection Stopped")

    def run(self):
        """Main application runner"""
        settings = self.render_sidebar()

        if st.sidebar.button("Start Threat Detection"):
            self.run_camera_detection(settings)

def main():
    app = ThreatDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
