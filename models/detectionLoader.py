from ultralytics import YOLO
def detectionModel():
    textmodel = YOLO(r"C:\Users\adgho\OneDrive\Desktop\ocr_regional\weights\best.pt")
    return textmodel