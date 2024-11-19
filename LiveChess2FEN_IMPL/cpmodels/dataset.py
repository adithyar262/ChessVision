"""This module works with the chess-piece dataset."""


import csv
import functools
import os
import shutil
from random import shuffle

import pandas as pd

from lc2fen.fen import PIECE_TYPES


PIECES_TO_CLASSNUM = {
    "_": 0,
    "b": 1,
    "k": 2,
    "n": 3,
    "p": 4,
    "q": 5,
    "r": 6,
    "B": 7,
    "K": 8,
    "N": 9,
    "P": 10,
    "Q": 11,
    "R": 12,
}


def randomize_dataset(dataset_dir):
    """Randomize the order of images in subdirectories of `dataset_dir`.

    The randomized images are renamed using the "<number>.jpg" format.

    Note that you should not use this on the downloaded dataset from
    LiveChess2FEN, as the dataset is already randomized.

    :param dataset_dir: Directory of the dataset.
    """
    dirs = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    for dir in dirs:
        files = os.listdir(dataset_dir + "/" + dir)
        shuffle(files)

        for i, file in enumerate(files):
            path = os.path.join(dataset_dir, dir, file)
            if os.path.isfile(path):
                newpath = os.path.join(dataset_dir, dir, str(i) + ".jpg")
                os.rename(path, newpath)


def split_dataset(
    dataset_dir: str = "../data/dataset/ChessPiecesDataset/",
    train_dir: str = "../data/dataset/train/",
    validation_dir: str = "../data/dataset/validation/",
    train_perc: float = 0.8,
):
    def safe_rmtree(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Ensure clean start
    safe_rmtree(train_dir)
    safe_rmtree(validation_dir)

    # Create necessary directories
    for base_dir in [train_dir, validation_dir]:
        os.makedirs(base_dir, exist_ok=True)
        for subdir in ["_/", "r/", "n/", "b/", "q/", "k/", "p/", "R/", "N/", "B/", "Q/", "K/", "P/"]:
            os.makedirs(base_dir + subdir, exist_ok=True)

    # Split dataset
    dirs = [
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    for dir in dirs:
        files = os.listdir(os.path.join(dataset_dir, dir))
        num_train_files = int(len(files) * train_perc)
        for i, file in enumerate(files):
            path = os.path.join(dataset_dir, dir, file)
            if os.path.isfile(path):
                newpath = (
                    os.path.join(train_dir, dir, file)
                    if i < num_train_files
                    else os.path.join(validation_dir, dir, file)
                )
                shutil.copy(path, newpath)
