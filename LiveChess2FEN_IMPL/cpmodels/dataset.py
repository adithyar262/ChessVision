"""This module works with the chess-piece dataset."""

import os
import shutil
from random import shuffle

# Mapping from piece symbols to descriptive class names
PIECES_TO_CLASSNAME = {
    "_": "empty_square",
    "black_bishop": "black_bishop",
    "black_king": "black_king",
    "black_knight": "black_knight",
    "black_pawn": "black_pawn",
    "black_queen": "black_queen",
    "black_rook": "black_rook",
    "white_bishop": "white_bishop",
    "white_king": "white_king",
    "white_knight": "white_knight",
    "white_pawn": "white_pawn",
    "white_queen": "white_queen",
    "white_rook": "white_rook",
}

def split_dataset(
    dataset_dir: str = "../data/dataset/ChessPiecesDataset_V2/ChessPiecesDataset_COMPLETE",
    train_dir: str = "../data/dataset/train/",
    validation_dir: str = "../data/dataset/validation/",
    train_perc: float = 0.8,
):
    def safe_rmtree(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Ensure a clean start
    safe_rmtree(train_dir)
    safe_rmtree(validation_dir)

    # Create necessary directories with new class names
    class_names = set(PIECES_TO_CLASSNAME.values())
    for base_dir in [train_dir, validation_dir]:
        os.makedirs(base_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

    # List all piece directories in the dataset
    piece_dirs = [
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for dir_symbol in piece_dirs:
        # Map the directory symbol to the class name
        class_name = PIECES_TO_CLASSNAME.get(dir_symbol)
        if class_name is None:
            print(f"Warning: Unrecognized directory '{dir_symbol}' in dataset.")
            continue

        # Get list of files and shuffle them
        files = os.listdir(os.path.join(dataset_dir, dir_symbol))
        shuffle(files)  # Shuffle for random splitting

        num_files = len(files)
        num_train = int(num_files * train_perc)
        train_files = files[:num_train]
        val_files = files[num_train:]

        # Copy files to the respective directories
        for file_name in train_files:
            src_path = os.path.join(dataset_dir, dir_symbol, file_name)
            dst_path = os.path.join(train_dir, class_name, file_name)
            shutil.copy(src_path, dst_path)

        for file_name in val_files:
            src_path = os.path.join(dataset_dir, dir_symbol, file_name)
            dst_path = os.path.join(validation_dir, class_name, file_name)
            shutil.copy(src_path, dst_path)

    print("Dataset splitting complete.")

if __name__ == "__main__":
    split_dataset()

