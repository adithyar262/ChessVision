import os

def create_piece_folders(base_dir: str):
    """Create folders for each chess piece type if they don't exist."""
    
    # All possible piece types including empty square
    piece_folders = ['_', 'b', 'B', 'k', 'K', 'n', 'N', 'p', 'P', 'q', 'Q', 'r', 'R']
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create individual piece folders
    for piece in piece_folders:
        piece_dir = os.path.join(base_dir, piece)
        if not os.path.exists(piece_dir):
            os.makedirs(piece_dir)
            print(f"Created folder for {piece}")
        else:
            print(f"Folder for {piece} already exists")

# Usage
base_dir = "data"
create_piece_folders(base_dir)

def organize_training_data_with_rotations(correct_fen: str, pieces_folder: str, output_base_dir: str):
    """
    Organize piece images and their rotations into training folders.
    
    :param correct_fen: The correct FEN string
    :param pieces_folder: Path to folder containing the 64 piece images
    :param output_base_dir: Base directory containing piece type folders
    """
    import os
    import shutil
    import cv2
    import numpy as np
    from datetime import datetime
    import glob
    
    def fen_to_board(fen: str) -> list:
        board = []
        for row in fen.split('/'):
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['_'] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        return board
    
    def rotate_image(image, angle):
        if angle == 0:
            return image
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Get sorted list of piece images
    piece_images = sorted(glob.glob(os.path.join(pieces_folder, "*.jpg")))
    if len(piece_images) != 64:
        raise ValueError(f"Expected 64 images, found {len(piece_images)}")
    
    # Convert FEN to 2D board
    board = fen_to_board(correct_fen.split()[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each piece image
    for idx, img_path in enumerate(piece_images):
        row = idx // 8
        col = idx % 8
        piece_type = board[row][col]
        dest_dir = os.path.join(output_base_dir, piece_type)
        
        try:
            # Read the original image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            
            # Generate and save rotations
            for angle in [0, 90, 180, 270]:
                rotated_img = rotate_image(img, angle)
                filename = f"piece_{timestamp}_{row}_{col}_rot{angle}.jpg"
                dest_path = os.path.join(dest_dir, filename)
                cv2.imwrite(dest_path, rotated_img)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Usage example:
correct_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
pieces_folder = "/home/adi/LiveChess2FEN/test_img/tmp/pieces"
output_base_dir = "data"

organize_training_data_with_rotations(correct_fen, pieces_folder, output_base_dir)

