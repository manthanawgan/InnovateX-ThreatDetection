# 🛡️ Trinetra AI Threat Detection System

A real-time video monitoring system that identifies potential security threats using computer vision and machine learning.

## 📋 Overview

Trinetra AI is a Streamlit web application that uses YOLO (You Only Look Once) machine learning models to detect threats in real-time video feeds. The system can detect different types of security concerns including:

- 🕵️ Potential intruders/thieves
- 🔫 Weapons (guns, knives, etc.)
- 🔍 Custom objects (using a customized model)

When threats are detected, the system triggers configurable alerts to notify security personnel.

## ⚙️ Features

- **Multiple Detection Modes**: Choose between Thief Detection, Weapon Detection, or Custom Detection
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity
- **Flexible Input Sources**: Support for default camera, IP camera, or uploaded videos
- **Configurable Warning System**:
  - Sound alerts
  - Visual alerts
  - Popup notifications
- **Real-time Status**: Live monitoring with detection count and status indicators
- **Annotated Video Feed**: Visual highlighting of detected objects

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Webcam or IP camera (for live detection)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trinetra-ai.git
cd trinetra-ai
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:
   - Place the default model (`best.pt`) in the root directory
   - Optional: Add your custom model (`my_model.pt`) for custom detections

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure settings in the sidebar:
   - Select detection mode
   - Adjust confidence threshold
   - Choose camera source
   - Select warning types

3. Click "Start Threat Detection" to begin monitoring

4. Press "Stop Detection" to end the session

## 📦 Required Packages

```
streamlit
opencv-python
numpy
ultralytics
pygame
scipy
```

## 🔄 Model Training

Trinetra AI uses YOLO models for object detection:

- The default model (`best.pt`) is trained to detect common security threats
- You can train and use your own custom model:
  1. Collect and label your custom dataset
  2. Train a YOLO model using Ultralytics framework
  3. Save the trained model as `my_model.pt` in the root directory

## 🔐 Privacy and Security Considerations

- All processing is done locally - no video data is sent to external servers
- Consider legal and ethical implications when deploying surveillance systems
- Ensure you have proper authorization before monitoring any area

## 🛠️ Troubleshooting

- **Camera not detected**: Verify webcam connections or IP camera address
- **Model loading errors**: Ensure model files are in the correct location
- **Sound alert issues**: Check pygame installation and audio device settings

## 📄 License

[MIT License](LICENSE)

## 👏 Acknowledgements

Trinetra AI utilizes the YOLO object detection framework by Ultralytics and is built with Streamlit for the web interface.

## 🔍 About the Name

"Trinetra" refers to the third eye in Hindu mythology, symbolizing higher consciousness and perception beyond ordinary sight - a fitting name for an AI-powered surveillance system designed to detect what might not be immediately visible to humans.