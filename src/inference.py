import cv2
from PIL import Image

from src.pipeline import run_detection_pipeline
from src.preprocess import preprocess_image
from models.loader import load_model


# load TrOCR model once
model, processor = load_model()


def run_inference(image_path):

    # 1. run detection pipeline
    lines = run_detection_pipeline(image_path)

    # 2. load image
    image = cv2.imread(image_path)

    final_text = []

    # 3. loop through lines
    for line in lines:

        # compute line bounding box
        x1 = min(box[0] for box in line)
        y1 = min(box[1] for box in line)
        x2 = max(box[2] for box in line)
        y2 = max(box[3] for box in line)

        # convert to int
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # optional padding (helps avoid cutting characters)
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        crop = image[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        # convert numpy → PIL
        pil_img = Image.fromarray(crop)

        # preprocessing
        processed_img = preprocess_image(pil_img)

        # TrOCR processing
        pixel_values = processor(processed_img, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        final_text.append(text)

    # combine lines
    result = "\n".join(final_text)

    return result