import os
import hashlib
from PIL import Image
import shutil

def get_image_hash(image_path):
    with Image.open(image_path) as img:
        img = img.resize((8, 8), Image.LANCZOS).convert('L')
        pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return ''.join('1' if pixel > avg else '0' for pixel in pixels)

def main():
    # Set the paths
    source_folder = 'img1'
    unique_folder = os.path.join(source_folder, 'unique_images')

    # Create the unique_images subfolder if it doesn't exist
    os.makedirs(unique_folder, exist_ok=True)

    # Dictionary to store image hashes
    image_hashes = {}
    unique_filenames = []

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(source_folder, filename)
            
            # Calculate the hash of the image
            image_hash = get_image_hash(file_path)
            
            # If the hash is not in our dictionary, it's a unique image
            if image_hash not in image_hashes:
                image_hashes[image_hash] = filename
                unique_filenames.append(filename)
                
                # Copy the unique image to the subfolder
                shutil.copy2(file_path, os.path.join(unique_folder, filename))

    # Print the list of unique filenames
    print("Unique image filenames:")
    for filename in unique_filenames:
        print(filename)

    print(f"\nTotal unique images found: {len(unique_filenames)}")
    print(f"Unique images have been copied to: {unique_folder}")

if __name__ == "__main__":
    main()
