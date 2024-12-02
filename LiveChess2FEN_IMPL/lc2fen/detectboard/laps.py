"""This is the lattice-points-search module."""

import collections
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist

from lc2fen.detectboard import debug, image_object
from lc2fen.detectboard import poly_point_isect

from lc2fen.detectboard.image_object import image_transform

__LAPS_SESS = onnxruntime.InferenceSession(
    "lc2fen/detectboard/models/laps_model.onnx"
)

__ANALYSIS_RADIUS = 10


def __find_intersections(lines):
    """Find all intersections."""
    __lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
    return poly_point_isect.isect_segments(__lines)


def __cluster_points(points, max_dist=10):
    """Cluster very similar points."""
    link_matrix = single(pdist(points))
    cluster_ids = fcluster(link_matrix, max_dist, "distance")

    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(points[i])
    clusters = clusters.values()
    # If two points are close, they become one mean point
    clusters = map(
        lambda arr: (
            np.mean(np.array(arr)[:, 0]),
            np.mean(np.array(arr)[:, 1]),
        ),
        clusters,
    )
    return list(clusters)


def __geometric_detector(img: np.ndarray) -> bool:
    """Determine if a point is a lattice point using geometric detection.

    :param img: Image to check.
    :return: True if it is a lattice point according to geometric detection.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)[1]
    img_edge = cv2.Canny(img_thresh, 0, 255)
    img_edge_resized = cv2.resize(img_edge, (21, 21), interpolation=cv2.INTER_CUBIC)

    # Geometric detector to filter easy points
    img_geo = cv2.dilate(img_edge_resized, None)
    mask = cv2.copyMakeBorder(
        img_geo,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    num_rhomboid = 0
    for cnt in contours:
        _, radius = cv2.minEnclosingCircle(cnt)
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and radius < 14:
            num_rhomboid += 1

    return num_rhomboid == 4


def __neural_detector(img_list: List[np.ndarray]) -> List[bool]:
    """Determine if points are lattice points using neural detector.

    :param img_list: List of images to check.
    :return: List of booleans indicating whether each image is a lattice point.
    """
    results = []
    for img in img_list:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)[1]
        img_edge = cv2.Canny(img_thresh, 0, 255)
        img_edge_resized = cv2.resize(img_edge, (21, 21), interpolation=cv2.INTER_CUBIC)
        img_binary = np.where(img_edge_resized > 127, 1, 0).astype("float32")

        # Reshape to match ONNX input requirements
        X = img_binary.reshape(1, 21, 21, 1)

        # Perform inference
        pred = __LAPS_SESS.run(None, {__LAPS_SESS.get_inputs()[0].name: X})[0][0]
        is_lattice = pred[0] > pred[1] and pred[1] < 0.03 and pred[0] > 0.975
        results.append(is_lattice)

    return results


def laps(img: np.ndarray, lines):
    intersection_points = __find_intersections(lines)

    debug.DebugImage(img).lines(lines, color=(0, 0, 255)).points(
        intersection_points, color=(255, 0, 0), size=2
    ).save("laps_in_queue")

    candidate_pts = []
    candidate_imgs = []
    valid_pts = []

    for pt in intersection_points:
        # Pixels are in integers
        pt = (int(pt[0]), int(pt[1]))

        if pt[0] < 0 or pt[1] < 0:
            continue

        # Size of our analysis area
        lx1 = max(0, pt[0] - __ANALYSIS_RADIUS - 1)
        lx2 = pt[0] + __ANALYSIS_RADIUS
        ly1 = max(0, pt[1] - __ANALYSIS_RADIUS)
        ly2 = pt[1] + __ANALYSIS_RADIUS + 1

        # Cropping for detector
        dimg = img[ly1:ly2, lx1:lx2]

        # Validate image size
        if dimg.size == 0:
            continue

        # Perform geometric detection
        if __geometric_detector(dimg):
            # It's a lattice point
            valid_pts.append(pt)
        else:
            # Need neural network detection
            candidate_pts.append(pt)
            candidate_imgs.append(dimg)

    if candidate_imgs:
        for img, pt in zip(candidate_imgs, candidate_pts):
            neural_result = __neural_detector([img])[0]
            if neural_result:
                valid_pts.append(pt)

    if valid_pts:
        valid_pts = __cluster_points(valid_pts)

    debug.DebugImage(img).points(
        intersection_points, color=(0, 0, 255), size=3
    ).points(valid_pts, color=(0, 255, 0)).save("laps_good_points")

    return valid_pts


def check_board_position(
    img: np.ndarray, board_corners: List[List[int]], tolerance: int = 20
) -> Tuple[bool, np.ndarray]:
    """Check if chessboard is in position given by the board corners.

    :param img: Image to check.
    :param board_corners: List of the coordinates of four board corners.
    :param tolerance: Number of lattice points that must be correct.
    :return: A tuple indicating if the chessboard is in position and the cropped image.
    """
    # Transform the image using the given board corners
    cropped_img = image_transform(img, board_corners)

    candidate_imgs = []
    for row_corner in range(150, 1200, 150):
        for col_corner in range(150, 1200, 150):
            # Size of our analysis area
            lx1 = max(0, row_corner - __ANALYSIS_RADIUS - 1)
            lx2 = row_corner + __ANALYSIS_RADIUS
            ly1 = max(0, col_corner - __ANALYSIS_RADIUS)
            ly2 = col_corner + __ANALYSIS_RADIUS + 1

            # Cropping for detector
            dimg = cropped_img[ly1:ly2, lx1:lx2]

            # Validate image size
            if dimg.size == 0:
                continue

            candidate_imgs.append(dimg)

    # Batch process candidate images through neural network
    neural_results = __neural_detector(candidate_imgs)
    correct_points = sum(neural_results)

    return correct_points >= tolerance, cropped_img
