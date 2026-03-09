import cv2
from models.detectionLoader import detectionModel

# Load YOLO detection model once
model = detectionModel()


# -------------------- DETECTION --------------------
def detect_boxes(image_path):
    results = model.predict(source=image_path, conf=0.5)
    boxes = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = box.conf[0].item()

            if confidence > 0.5:
                boxes.append(xyxy)

    return boxes


# -------------------- SORTING --------------------
def sort_boxes(boxes):
    # sort top → bottom
    return sorted(boxes, key=lambda b: b[1])


# -------------------- GROUPING --------------------
def group_boxes_into_lines(boxes, y_threshold=30):
    lines = []

    for box in boxes:
        x1, y1, x2, y2 = box
        placed = False

        for line in lines:
            _, ly1, _, _ = line[0]

            if abs(y1 - ly1) < y_threshold:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    return lines


# -------------------- SORT WORDS IN EACH LINE --------------------
def sort_lines(lines):
    # sort left → right
    return [sorted(line, key=lambda b: b[0]) for line in lines]


# -------------------- DETECTION PIPELINE --------------------
def run_detection_pipeline(image_path):
    boxes = detect_boxes(image_path)
    boxes = sort_boxes(boxes)

    lines = group_boxes_into_lines(boxes)
    lines = sort_lines(lines)

    return lines


# -------------------- CROPPING --------------------
def crop_box(image, box):
    x1, y1, x2, y2 = map(int, box)

    # safety checks
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop