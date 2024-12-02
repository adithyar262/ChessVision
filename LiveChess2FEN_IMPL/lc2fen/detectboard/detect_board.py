"""This is the main file of detectboard module.

It detects a board on a given image using the `detect()` function.
"""


import cv2
import numpy as np
import logging

from lc2fen.detectboard import debug
from lc2fen.detectboard.cps import cps
from lc2fen.detectboard.image_object import ImageObject
from lc2fen.detectboard.laps import laps, check_board_position
from lc2fen.detectboard.slid import slid

from lc2fen.detectboard.image_object import image_transform


def __original_points_coords(point_list, border):
    """Detect the coordinates of the board in the original image.

    :param point_list: List of the relative points.

    The relative points are in the sequence of image transformations
    done in each layer.

    :return: The coordinates in the original image of the chessboard
    corners and the coordinates of each of the corners of the
    chessboard squares as a pair of `board_corners` and
    `square_corners`.
    """
    ptslims = np.float32([
        [border, border],
        [1200 - border, border],
        [1200 - border, 1200 - border],
        [border, 1200 - border]
    ])
    last_index = len(point_list) - 1

    # Compute all the transformation matrices
    transf_mats = []
    for i in range(last_index):
        transf_mat = cv2.getPerspectiveTransform(
            np.float32(point_list[i]), ptslims
        )
        cv2.invert(transf_mat, transf_mat)
        transf_mats.append(transf_mat)

    # Multiply into an equivalent single transformation matrix
    transf_mat = transf_mats[0]
    for i in range(1, last_index):
        transf_mat = transf_mat.dot(transf_mats[i])

    # Transform the actual corner points
    transf_points = cv2.perspectiveTransform(
        np.float32(point_list[last_index]).reshape(-1, 1, 2), transf_mat
    )

    board_corners = np.int32(
        [transf_points[i][0] for i in range(len(transf_points))]
    )

    # Now obtain the corners of each square in the chessboard
    # To do so we need also the last transformation matrix
    last_transf_mat = cv2.getPerspectiveTransform(
        np.float32(point_list[last_index]), ptslims
    )
    cv2.invert(last_transf_mat, last_transf_mat)
    transf_mat = transf_mat.dot(last_transf_mat)

    # Generate the corners of the squares with the new dimensions
    corners = []
    for row_corner in range(0, 1200 + 2 * border + 150, 150):
        for col_corner in range(0, 1200 + 2 * border + 150, 150):
            corners.append([row_corner, col_corner])

    # Transform the corners of the squares
    square_corners = cv2.perspectiveTransform(
        np.float32(corners).reshape(-1, 1, 2), transf_mat
    )

    square_corners = np.int32(
        [square_corners[i][0] for i in range(len(square_corners))]
    )

    return board_corners, square_corners


def __layer(img):
    """Execute one layer (iteration) on the given image."""
    # Step 1 --- Straight line detector
    lines = slid(img["main"])

    # Step 2 --- Lattice points search
    points = laps(img["main"], lines)

    # Step 3 --- Chessboard position search
    four_points = cps(img["main"], points, lines)

    # Crop the image for the next step
    img.crop(four_points)


def detect(
    input_image: np.ndarray,
    output_board: str,
    board_corners: (list[list[int]] | None) = None,
):
    # Check if we can skip full board detection (if board position is already known)
    if board_corners is not None:
        found, cropped_img = check_board_position(input_image, board_corners)
        if found:
            cv2.imwrite(output_board, cropped_img)
            image = ImageObject(input_image)
            # For corners calculation
            image.add_points([[0, 0], [1200, 0], [1200, 1200], [0, 1200]])
            image.add_points(board_corners)
            return image

    # Read the input image and store the cropped detected board
    n_layers = 3
    image = ImageObject(input_image)
    for i in range(n_layers):
        __layer(image)
        debug.DebugImage(image["orig"]).save(f"end_iteration{i}")
    # cv2.imwrite(output_board, image["orig"])

    return image


def compute_corners(image_object, border=10):
    board_corners, square_corners = __original_points_coords(
        image_object.get_points(), border=border
    )

    # Adjust the output size to include the border
    output_size = (1200 + 2 * border, 1200 + 2 * border)

    # Transform the original image using board_corners
    original_image = image_object.get_images()[0]["orig"]
    corrected_board_image = image_transform(original_image, board_corners, output_size, border)

    debug.DebugImage(original_image).points(
        square_corners, size=50, color=(0, 0, 255)
    ).points(board_corners, size=50, color=(0, 255, 0)).save("corner_points")

    return board_corners, corrected_board_image
