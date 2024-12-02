import shutil
import glob

import numpy as np
import cv2
import os
import chess
import onnxruntime

from keras.api.models import load_model
from keras.api.utils import load_img, img_to_array
from lc2fen.detectboard.detect_board import detect, compute_corners
from lc2fen.split_board import split_board_image_trivial
from lc2fen.infer_pieces import infer_chess_pieces

from lc2fen.fen import (
    list_to_board,
    board_to_fen,
    compare_fen,
    is_light_square,
    fen_to_board,
    board_to_list,
)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
except ImportError:
    cuda = None
    trt = None


def load_image(img_path: str, img_size: int, preprocess_func) -> np.ndarray:
    img = load_img(img_path, target_size=(img_size, img_size))
    img = img.convert('RGB')  # Ensure image has 3 channels
    img_tensor = img_to_array(img)
    img_tensor = preprocess_func(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor


def predict_board_keras(
        model_path: str,
        img_size: int,
        pre_input,
        path="",
        a1_pos="",
        test=False,
        previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]] | None:
    # Load the model
    model = load_model(model_path)

    def obtain_piece_probs_for_all_64_squares(
            pieces: list[str],
    ) -> list[list[float]]:
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            # Predict the piece, pick the first prediction
            predictions.append(model.predict(piece_img)[0])
        return predictions

    # Predict a single image
    return predict_board(
        path,
        a1_pos,
        obtain_piece_probs_for_all_64_squares,
        previous_fen=previous_fen,
    )


def predict_board_onnx(
        model_path: str,
        img_size: int,
        pre_input,
        path="",
        a1_pos="",
        test=False,
        previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]] | None:
    sess = onnxruntime.InferenceSession(model_path)

    def obtain_piece_probs_for_all_64_squares(
            pieces: list[str],
    ) -> list[list[float]]:
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, img_size, pre_input)
            predictions.append(
                sess.run(None, {sess.get_inputs()[0].name: piece_img})[0][0]
            )
        return predictions

    return predict_board(
        path,
        a1_pos,
        obtain_piece_probs_for_all_64_squares,
        previous_fen=previous_fen,
    )


def predict_board_trt(
        model_path: str,
        img_size: int,
        pre_input,
        path="",
        a1_pos="",
        test=False,
        previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]] | None:
    if cuda is None or trt is None:
        raise ImportError("Unable to import pycuda or tensorrt")

    class __HostDeviceTuple:
        """A tuple of host and device. It helps clarify code."""

        def __init__(self, _host, _device):
            self.host = _host
            self.device = _device

    def __allocate_buffers(engine, batch_size):
        """Allocate all buffers required for the specified engine."""
        inputs = []
        outputs = []
        bindings = []

        binding_index = 0
        while True:
            try:
                binding_name = engine.get_tensor_name(binding_index)
                shape = engine.get_tensor_shape(binding_name)
                shape[0] = batch_size if shape[0] == -1 else shape[0]
                size = trt.volume(shape)
                dtype = trt.nptype(engine.get_tensor_dtype(binding_name))

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                bindings.append(int(device_mem))

                if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                    inputs.append(__HostDeviceTuple(host_mem, device_mem))
                else:
                    outputs.append(__HostDeviceTuple(host_mem, device_mem))
                binding_index += 1
            except Exception:
                break

        stream = cuda.Stream()
        return inputs, outputs, bindings, stream

    def __infer(context, bindings, inputs, outputs, stream):
        """Infer outputs on IExecutionContext for specified inputs."""
        for binding_idx, inp in enumerate(inputs):
            tensor_name = engine.get_tensor_name(binding_idx)  # Use get_tensor_name
            context.set_tensor_address(tensor_name, inp.device)
        for binding_idx, out in enumerate(outputs, start=len(inputs)):
            tensor_name = engine.get_tensor_name(binding_idx)  # Use get_tensor_name
            context.set_tensor_address(tensor_name, out.device)

        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)

        context.execute_async_v3(stream_handle=stream.handle)

        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        stream.synchronize()
        return [out.host for out in outputs]

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Set valid batch size
    batch_size = 16
    inputs, outputs, bindings, stream = __allocate_buffers(engine, batch_size)

    # Allocate input data
    input_shape = (batch_size, img_size, img_size, 3)
    img_array = np.zeros(input_shape, dtype=np.float32)

    with engine.create_execution_context() as context:
        # Get the tensor name for the first binding (index 0)
        input_name = engine.get_tensor_name(0)

        # Check if the tensor has a dynamic shape
        if -1 in engine.get_tensor_shape(input_name):
            context.set_input_shape(input_name, input_shape)

        def obtain_piece_probs_for_all_64_squares(pieces: list[str]) -> list[list[float]]:
            for i, piece in enumerate(pieces):
                img_array[i % batch_size] = load_image(piece, img_size, pre_input)
                if (i + 1) % batch_size == 0 or i == len(pieces) - 1:
                    np.copyto(inputs[0].host, img_array.ravel())  # Flatten data for copying
                    trt_outputs = __infer(context, bindings, inputs, outputs, stream)[-1]
                    for ind in range(0, len(trt_outputs), 13):
                        yield trt_outputs[ind: ind + 13]

        # Convert generator to a list within `predict_board_trt`
        probs_with_no_indices = list(obtain_piece_probs_for_all_64_squares(obtain_individual_pieces(path)))

        if test:
            test_predict_board(probs_with_no_indices)
        else:
            return predict_board(
                path, a1_pos, lambda _: probs_with_no_indices, previous_fen=previous_fen
            )


def predict_board(
        board_path: str,
        a1_pos: str,
        obtain_piece_probs_for_all_64_squares,
        board_corners: (list[list[int]] | None) = None,
        previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]]:
    board_corners = detect_input_board(board_path, board_corners)
    pieces = obtain_individual_pieces(board_path)
    probs_with_no_indices = obtain_piece_probs_for_all_64_squares(pieces)

    if previous_fen is not None and not check_validity_of_fen(previous_fen):
        print("Warning: the previous FEN is ignored because it is invalid for a standard physical chess set")
        previous_fen = None
    predictions = infer_chess_pieces(
        probs_with_no_indices, a1_pos, previous_fen
    )

    board = list_to_board(predictions)
    fen = board_to_fen(board)

    return fen, board_corners


def detect_input_board(board_path: str, board_corners: (list[list[int]] | None) = None) -> list[list[int]]:
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


def obtain_individual_pieces(board_path: str) -> list[str]:
    head, tail = os.path.split(board_path)
    tmp_dir = os.path.join(head, "tmp/")
    pieces_dir = os.path.join(tmp_dir, "pieces/")
    os.mkdir(pieces_dir)
    split_board_image_trivial(os.path.join(tmp_dir, tail), "", pieces_dir)
    return sorted(glob.glob(pieces_dir + "/*.jpg"))


def check_validity_of_fen(fen: str) -> bool:
    board = chess.Board(fen)
    if not board.is_valid():  # If it's white to move, the FEN is invalid
        board.turn = chess.BLACK
        if (
                not board.is_valid()
        ):  # If it's black to move, the FEN is also invalid
            return False

    num_of_P = fen.count("P")  # Number of white pawns
    num_of_Q = fen.count("Q")  # Number of white queens
    num_of_R = fen.count("R")  # Number of white rooks
    num_of_N = fen.count("N")  # Number of white knights
    num_of_p = fen.count("p")  # Number of black pawns
    num_of_q = fen.count("q")  # Number of black queens
    num_of_r = fen.count("r")  # Number of black rooks
    num_of_n = fen.count("n")  # Number of black knights
    fen_list = board_to_list(fen_to_board(fen))
    num_of_light_squared_B = sum(
        [
            is_light_square(square)
            for (square, piece_type) in enumerate(fen_list)
            if piece_type == "B"
        ]
    )  # Number of light-squared bishops for white
    num_of_dark_squared_B = (
            fen.count("B") - num_of_light_squared_B
    )  # Number of dark-squared bishops for white
    num_of_light_squared_b = sum(
        [
            is_light_square(square)
            for (square, piece_type) in enumerate(fen_list)
            if piece_type == "b"
        ]
    )  # Number of light-squared bishops for black
    num_of_dark_squared_b = (
            fen.count("b") - num_of_light_squared_b
    )  # Number of dark-squared bishops for black

    if (
            num_of_R > 2
            or num_of_r > 2
            or num_of_N > 2
            or num_of_n > 2
            or (num_of_light_squared_B + num_of_dark_squared_B) > 2
            or (num_of_light_squared_b + num_of_dark_squared_b) > 2
            or num_of_Q > 2
            or num_of_q > 2
    ):  # Number of any piece is too large for a standard physical chess set
        return False

    if (
            num_of_P == 7
            and num_of_Q == 2  # A white pawn has promoted into a queen
            and (
            num_of_light_squared_B == 2 or num_of_dark_squared_B == 2
    )  # A white pawn has promoted into a bishop
    ):
        return False

    if num_of_P == 8 and (
            num_of_Q == 2  # A white pawn has promoted into a queen
            or (
                    num_of_light_squared_B == 2 or num_of_dark_squared_B == 2
            )  # A white pawn has promoted into a bishop
    ):
        return False

    if (
            num_of_p == 7
            and num_of_q == 2  # A black pawn has promoted into a queen
            and (
            num_of_light_squared_b == 2 or num_of_dark_squared_b == 2
    )  # A black pawn has promoted into a bishop
    ):
        return False

    if num_of_p == 8 and (
            num_of_q == 2  # A black pawn has promoted into a queen
            or (
                    num_of_light_squared_b == 2 or num_of_dark_squared_b == 2
            )  # A black pawn has promoted into a bishop
    ):
        return False

    return True
