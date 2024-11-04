from PIL import Image
import numpy as np

# Load the image
image = Image.open('../textures/board_complete.png')  # Replace with your image file name

# Convert the image to RGBA (to handle transparency)
image = image.convert('RGBA')

# Convert image data to a NumPy array for easier manipulation
data = np.array(image)

# Define color ranges for white and green squares (based on typical chessboard colors)
# Adjust these values if necessary based on the exact shades of green and white in your image
white_square = [240, 240, 240]  # Approximate RGB for white squares
green_square = [118, 150, 86]   # Approximate RGB for green squares

# Create a mask that identifies pixels that are not part of the background (i.e., not green or white)
mask = ~(
    ((np.abs(data[:, :, :3] - white_square) < 30).all(axis=-1)) |  # White square threshold
    ((np.abs(data[:, :, :3] - green_square) < 30).all(axis=-1))    # Green square threshold
)

# Create a new array for the output image with transparent background
new_data = np.zeros_like(data)

# Copy over only the pixels corresponding to chess pieces (where mask is True)
new_data[mask] = data[mask]

# Set non-piece areas to fully transparent (alpha = 0)
new_data[~mask, 3] = 0

# Convert back to an image
new_image = Image.fromarray(new_data, 'RGBA')

# Save the new image with transparent background
new_image.save('../textures/chess_pieces_transparent.png')
