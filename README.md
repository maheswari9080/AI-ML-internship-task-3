# ♻️ Waste Classification AI System (Plastic, Paper, Metal)
### AI/ML Internship — Task 3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-105%20Images-blue.svg)](#-dataset-overview)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](#-model-details)
[![Studio](https://img.shields.io/badge/Studio-Interactive%20Web%20Viewer-green.svg)](#-interactive-dataset-studio)

---

## 📌 Project Overview
This repository contains the complete dataset, trained deep learning models, interactive dataset curation studio, and technical documentation for **Task 3: Automated Waste Classification**.

The system classifies recyclable waste into three primary categories:
- **🧴 Plastic Waste** (Bottles, containers, covers, jugs)
- **📰 Paper Waste** (Cardboards, newspapers, office papers, boxes)
- **🥫 Metal Waste** (Beverage cans, tin boxes, metal scraps, caps)

---

## 📁 Repository Structure

```text
├── metal/                      # Metal waste image samples (35 images)
│   ├── metal_01.jpg ... metal_35.jpg
├── paper/                      # Paper waste image samples (35 images)
│   ├── paper_01.jpg ... paper_35.jpg
├── plastic/                    # Plastic waste image samples (35 images)
│   ├── plastic_01.jpg ... plastic_35.jpg
├── converted_keras.zip         # Trained Keras model package (keras_model.h5 + labels.txt)
├── mw_dataset_viewer.html      # Interactive Dataset Viewer & Webcam Capture Studio
├── Task 3 MW.docx              # Comprehensive Task Report & Experimentation Document
├── .gitignore                  # Git ignore rules for system & temp files
├── LICENSE                     # MIT Open Source License
└── README.md                   # Project documentation
```

---

## 📊 Dataset Overview

The dataset was curated with diverse real-world object conditions to maximize model robustness and generalization:

| Class | Count | Key Variations Captured |
|---|---|---|
| **🧴 Plastic** | 35 | Bottles, cups, food containers, varied lighting & angles |
| **📰 Paper** | 35 | Folded newspapers, corrugated cardboard, white sheets, crushed paper |
| **🥫 Metal** | 35 | Soda cans, food tins, metallic wrappers, different orientations |
| **Total** | **105** | **Multi-angle, varied distances (close-up to far), complex backgrounds** |

### Image Variation Checklist
To prevent overfitting and handle real-world deployment challenges:
- **Object Angles**: Tilted, 45°, top-down, and side-view.
- **Distances**: Close-ups to full-frame object views.
- **Backgrounds**: Neutral solid surfaces, cluttered desktop backdrops, textured floors.
- **Lighting Conditions**: Direct indoor lighting, ambient daylight, and slight shadows.

---

## 🌐 Interactive Dataset Studio & Webcam Studio

The included [`mw_dataset_viewer.html`](mw_dataset_viewer.html) provides a modern web interface with zero installation required:

1. **Dataset Gallery**:
   - Visual inspection of all 105 samples categorized into Plastic, Paper, and Metal.
   - Quick counters and responsive cards with preview zoom.

2. **Webcam Capture Studio**:
   - Real-time video stream for collecting new dataset samples.
   - **Variation Guidance Prompts**: In-browser checklist for camera angles, backgrounds, and orientations.
   - **Burst Mode (5x)**: Quick collection of continuous sample frames.
   - **1-Click Batch Export**: Download all session captures directly as labeled `.jpg` files.

### How to Run:
Double-click [`mw_dataset_viewer.html`](mw_dataset_viewer.html) to open it in any modern browser (Chrome, Edge, Firefox, Safari).

---

## 🤖 Model Details

- **Architecture**: Transfer Learning MobileNet / Teachable Machine Deep Neural Network
- **Export Format**: Keras H5 (`converted_keras.zip` contains `keras_model.h5` and `labels.txt`)
- **Input Resolution**: `224x224x3` RGB normalized images
- **Output Classes**:
  ```text
  0 Plastic
  1 Paper
  2 Metal
  ```

### Quick Python Inference Example

```python
import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageOps

# Load trained model and class labels
model = keras.models.load_model("keras_model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Preprocess image
image = Image.open("plastic/plastic_01.jpg").convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)

# Normalize image to [-1, 1]
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# Predict class
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print(f"Prediction: {class_name} | Confidence: {confidence_score:.2%}")
```

---

## 🔄 Real-World Application & Architecture

This classifier is designed for integration into smart environmental management systems:

```text
  [ Camera / Sensor ]
          │
          ▼
   [ Waste Image ]
          │
          ▼
  [ Deep Learning Model ] (TensorFlow / Keras)
          │
          ▼
 [ Category Classification ] ──▶ (Plastic / Paper / Metal)
          │
          ▼
[ Smart Actuator / Sorting Bin ] ──▶ Automated Robotic Sorting & Recycling
```

### Potential Applications:
- **Smart Municipal Waste Bins**: Automatic door-opening for designated recyclables.
- **Conveyor Belt Sorting Facilities**: High-speed automated sorting at recycling plants.
- **Citizen Recycling Assistive Apps**: Mobile assistant guiding households on proper waste segregation.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
