def detect(
    input_image: np.ndarray,
    output_board: str,
    board_corners: (list[list[int]] | None) = None,
):
    """Detect the board position and store the cropped detected board.

    This function detects the board position in `input_image` and stores
    the cropped detected board in `output_board`.

    :param input_image: Input chessboard image.

    :param output_board: Output path for the detected-board image.

        This path must include both the name and extension.

    :param board_corners: List of coordinates of the four board corners.

        If it is not None, first check if the board is in the position
        given by these corners. If not, runs the full detection.

    :return: Final ImageObject with which to compute the corners if
    necessary.
    """
    # Check if we can skip full board detection (if board position is
    # already known)
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
    cv2.imwrite(output_board, image["orig"])

    return image

def detect_input_board(
    board_path: str, board_corners: (list[list[int]] | None) = None
) -> list[list[int]]:
    """Detect the input board.

    This function takes as input a path to a chessboard image
    (e.g., "image.jpg") and stores the image that contains the detected
    chessboard in the "tmp" subfolder of the folder containing the board
    (e.g., "tmp/image.jpg").

    If the "tmp" folder already exists, the function deletes its
    contents. Otherwise, the function creates the "tmp" folder.

    :param board_path: Path to the chessboard image of interest.

        The path must have rw permission.

        Example: `"../data/predictions/board.jpg"`.

    :param board_corners: Length-4 list of coordinates of four corners.

        The 4 board corners are in the order of top left, top right,
        bottom right, and bottom left.

        If it is not `None` and the corner coordinates are accurate
        enough, the neural-network-based board-detection step is skipped
        (which means the total processing time is reduced).

    :return: Length-4 list of the (new) coordinates of the four board
    corners detected.
    """
    input_image = cv2.imread(board_path)
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    image_object = detect(
        input_image, os.path.join(head, "tmp", tail), board_corners
    )
    board_corners, _ = compute_corners(image_object)
    return board_corners