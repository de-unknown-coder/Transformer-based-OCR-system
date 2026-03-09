# Transformer-YOLO OCR Pipeline
### Extending Transformer-based OCR to Paragraph-Level Text

## Overview

This project presents a modular Optical Character Recognition (OCR) pipeline designed to extend transformer-based OCR models from **single-line text recognition to paragraph-level text recognition**.

Most transformer OCR models such as **TrOCR** are trained on **single-line text images**, while real-world documents often contain **multi-line paragraphs**. Training transformers directly on paragraph-level images requires large datasets and complex layout understanding.

To address this limitation, this project integrates a **YOLO-based text detection model** with a **transformer-based OCR model**. The detection model identifies text regions in an image, groups them into lines, and passes each line to the OCR model for recognition.

The final recognized lines are combined to reconstruct the full paragraph.

---

## Project Goal

The main goal of this project is to develop a **paragraph-level OCR system capable of supporting multiple languages**.

The architecture separates:

- **Text Detection**
- **Text Recognition**

This modular design allows each component to be improved independently.

---

## System Architecture

```
Input Image
     ↓
YOLO Text Detection
     ↓
Bounding Box Sorting
     ↓
Line-level Cropping
     ↓
Transformer OCR (TrOCR)
     ↓
Paragraph Reconstruction
     ↓
Final Text Output
```

---

## Pipeline Steps

### 1. Text Detection
A YOLO-based model detects text regions in the image.

### 2. Line Grouping
Detected bounding boxes are sorted and grouped into lines based on spatial coordinates.

### 3. Line-level Cropping
Each line is cropped from the image to preserve contextual information.

### 4. Transformer-based Recognition
Each cropped line is processed using the TrOCR model to generate the corresponding text.

### 5. Paragraph Reconstruction
Recognized lines are combined to reconstruct the full paragraph.

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- OpenCV
- YOLO Text Detection
- TrOCR (Transformer OCR)

---

## Current Features

- Paragraph-level OCR pipeline
- YOLO-based text detection
- Line segmentation
- Transformer-based text recognition
- Context-aware recognition using line crops
- Modular architecture

---

## Training Setup

The OCR model is fine-tuned using **line-level text images**.

Training configuration:

```
Model: TrOCR (pretrained)
Dataset: Line-level text images
Optimizer: AdamW
Learning Rate: 3e-5 – 5e-5
Epochs: 20+
Mixed Precision Training
```

---

## Current Results

The system can:

- Detect multi-line text from images
- Segment text into lines
- Recognize paragraph-level text using transformer OCR

Example:

```
Input Image → Recognized Paragraph Text
```

---

## Remaining Tasks

The current implementation focuses on **English OCR and pipeline development**. The following improvements are planned:

### 1. Multilingual OCR Support
Fine-tune the model for additional scripts such as:

- Hindi (Devanagari)
- Other regional languages

### 2. Mixed-language Training
Train the model on **combined multilingual datasets** so that one model can recognize multiple scripts.

### 3. Script Detection
Automatically detect which language/script is present in the image.

### 4. Detection Improvements
Improve YOLO text detection for complex document layouts.

### 5. API / Web Interface
Develop a simple interface where users can upload images and receive OCR results.

---

## Repository Structure

```
ocr-project/
│
├── api/
├── configs/
├── datasets/
├── models/
│   ├── loader.py
│   ├── detectionLoader.py
│
├── src/
│   ├── pipeline.py
│   ├── inference.py
│   ├── preprocess.py
│   ├── test_inference.py
│
├── web/
│
├── weights/
│   ├── trocr_english/
│   ├── trocr_hindi/
│
├── README.md
└── requirements.txt
```

---

## Future Work

The long-term objective is to build a **multilingual paragraph-level OCR system** capable of recognizing multiple regional languages using a unified transformer architecture.

Possible future extensions include:

- multilingual transformer training
- document layout understanding
- handwriting OCR
- large-scale OCR dataset integration

---

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017  
2. Li et al., *TrOCR: Transformer-based OCR with Pretrained Models*, 2021  
3. Redmon et al., *You Only Look Once: Unified Real-Time Object Detection*, CVPR 2016  
4. PaddleOCR GitHub Repository