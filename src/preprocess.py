from PIL import Image
from configs.configs import image_size

def preprocess_image(image : Image.Image):
    image = image.convert("RGB")
    image = image.resize(image_size)
    
    return image
