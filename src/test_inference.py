from src.inference import run_inference
from configs.configs import sample_path


def main():

    image_path = sample_path

    result = run_inference(image_path)

    print("\n===== OCR OUTPUT =====\n")
    print(result)


if __name__ == "__main__":
    main()