from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model
model = YOLO(r"C:\Users\adgho\OneDrive\Desktop\ocr_regional\weights\best.pt")

# Image path
img_path = r"C:\Users\adgho\Downloads\english2.jpg"

# Load image using OpenCV
image = cv2.imread(img_path)

# Run inference
results = model.predict(
    source=img_path,
    conf=0.25,
    iou=0.7,
    imgsz=640
)

# Draw boxes
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        confidence = box.conf[0].item()

        x1, y1, x2, y2 = map(int, xyxy)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put confidence text
        cv2.putText(
            image,
            f"{confidence:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

# Convert BGR → RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("YOLO Text Detection")
plt.axis("off")
plt.show()