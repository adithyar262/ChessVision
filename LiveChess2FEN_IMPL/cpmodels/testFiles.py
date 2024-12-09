
import os
from PIL import Image


def verify_images(directory):
    invalid_images = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError) as e:
                    print(f"Invalid image: {file_path}")
                    invalid_images.append(file_path)
    return invalid_images


if __name__ == "__main__":
    train_dir = "../data/dataset/train/"
    validation_dir = "../data/dataset/validation/"

    print("Checking training images...")
    bad_train_images = verify_images(train_dir)
    print(f"Found {len(bad_train_images)} invalid training images.")

    print("Checking validation images...")
    bad_validation_images = verify_images(validation_dir)
    print(f"Found {len(bad_validation_images)} invalid validation images.")

    if bad_train_images or bad_validation_images:
        print("Please remove or fix the invalid images before retrying training.")