"""This module is responsible for iteratively finding a chessboard."""


import math
from typing import Tuple, Any

import cv2
import numpy as np
from cv2 import Mat
from numpy import ndarray, dtype


def image_scale(pts, scale):
    """Scale to original image size."""
    return [[x / scale, y / scale] for (x, y) in pts]


def image_resize(img: np.ndarray, target_area: int = 250000) -> tuple[ndarray | Any, tuple[int, ...], float]:
    shape = img.shape
    current_area = shape[0] * shape[1]
    scale = math.sqrt(target_area / current_area)
    new_dimensions = (int(shape[1] * scale), int(shape[0] * scale))
    resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_img, shape, scale


def image_transform(img: np.ndarray, points, output_size=(1200, 1200), border=0):
    """Crop original image using perspective warp."""
    # Include the border in the destination points
    pts1 = np.float32(points)
    pts2 = np.float32([
        [border, border],  # Top-left corner with border
        [output_size[0] - border, border],  # Top-right corner with border
        [output_size[0] - border, output_size[1] - border],  # Bottom-right corner
        [border, output_size[1] - border]  # Bottom-left corner
    ])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, mat, output_size)


class ImageObject:
    """Represent an image object in process of finding chessboard.

    This class represents an image object in the iterative process of
    finding a chessboard.
    """

    def __init__(self, img: (np.ndarray | None) = None):
        """Save and prepare image array."""
        # We save the whole sequence of transformations attribute[i] is
        # the attribute of iteration i, with iteration 0 being the first
        # one
        self.points = []  # Points of the new cropped image for next iteration
        self.images = []
        self.shape = []  # (0, 0)
        self.scale = []  # 1
        if img is not None:
            # Downscale for speed
            downscaled_img_, shape_, scale_ = image_resize(img)

            self.images.append({"orig": img, "main": downscaled_img_})
            self.shape.append(shape_)  # (0, 0)
            self.scale.append(scale_)  # 1

    def __getitem__(self, attr):
        """Return last image as array."""
        return self.images[-1][attr]

    def __setitem__(self, attr, val):
        """Save image to object as last image."""
        self.images[-1][attr] = val

    def add_image(self, img: np.ndarray):
        """Add a new image in the iteration."""
        # Downscale for speed
        downscaled_img_, shape_, scale_ = image_resize(img)

        self.images.append({"orig": img, "main": downscaled_img_})
        self.shape.append(shape_)
        self.scale.append(scale_)

    def crop(self, pts):
        """Crop using 4 points transform."""
        pts_orig = image_scale(pts, self.scale[-1])
        img_crop = image_transform(self.images[-1]["orig"], pts_orig)
        self.points.append(pts_orig)
        self.add_image(img_crop)

    def add_points(self, points):
        """Add points to the point list."""
        self.points.append(points)

    def get_images(self):
        """Return images list."""
        return self.images

    def get_points(self):
        """Return points list."""
        return self.points
