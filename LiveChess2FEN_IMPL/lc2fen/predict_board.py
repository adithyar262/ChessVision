import shutil
import glob
from typing import Tuple, Any
from typing import List

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


def load_image(img_array: np.ndarray, img_size: int, preprocess_func) -> np.ndarray:
    # Resize the image
    img = cv2.resize(img_array, (img_size, img_size))
    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to array and preprocess
    img_tensor = img_to_array(img)
    img_tensor = preprocess_func(img_tensor)
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

    def obtain_piece_probs_for_all_64_squares(pieces: List[np.ndarray]) -> List[List[float]]:
        # Preprocess all pieces
        piece_imgs = [pre_input(cv2.resize(piece, (img_size, img_size))) for piece in pieces]
        batch_input = np.array(piece_imgs)

        # Predict in batch
        predictions = model.predict(batch_input)
        return predictions.tolist()

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

    def obtain_piece_probs_for_all_64_squares(pieces: List[np.ndarray]) -> List[List[float]]:
        # Preprocess all pieces
        piece_imgs = [load_image(piece, img_size, pre_input) for piece in pieces]
        batch_input = np.array(piece_imgs).astype(np.float32)
        # Ensure correct input shape
        if len(batch_input.shape) == 3:
            batch_input = np.expand_dims(batch_input, axis=0)
        # Predict in batch
        predictions = sess.run(None, {sess.get_inputs()[0].name: batch_input})[0]
        return predictions.tolist()

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
) -> Tuple[str, List[List[int]]]:
    if cuda is None or trt is None:
        raise ImportError("Unable to import pycuda or tensorrt")

    class HostDeviceMem:
        """Simple class to encapsulate host and device memory"""

        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(engine, context):
        """
        Allocates all buffers required for an engine using the new API.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            # Get dynamic tensor shape from context
            shape = context.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to device bindings
            bindings.append(int(device_mem))

            # Append to appropriate list (input or output)
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def run_inference(context, bindings, inputs, outputs, stream, batch_size):
        """
        Executes inference using the provided context and buffers.
        """
        # Transfer input data to device
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)

        # Set tensor addresses for all bindings
        for i in range(len(bindings)):
            tensor_name = context.engine.get_tensor_name(i)
            context.set_tensor_address(tensor_name, bindings[i])

        # Execute inference
        context.execute_async_v3(stream_handle=stream.handle)

        # Transfer output data back to host
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        # Synchronize the stream
        stream.synchronize()

        # Return the output host memory
        return [out.host for out in outputs]

    # Load TensorRT engine
    with open(model_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(f.read())

    # Create execution context
    context = engine.create_execution_context()

    # Set batch size
    batch_size = 16

    # Prepare input shape (assuming NHWC format)
    input_shape = (batch_size, img_size, img_size, 3)

    # Set the binding shape dynamically for the input tensor
    input_tensor_name = engine.get_tensor_name(0)
    context.set_input_shape(input_tensor_name, input_shape)

    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

    def obtain_piece_probs_for_all_64_squares(pieces: List[np.ndarray]) -> List[List[float]]:
        """
        Process pieces in batches and run inference.
        """
        # Preprocess all pieces
        piece_imgs = [load_image(piece, img_size, pre_input) for piece in pieces]
        batch_input = np.array(piece_imgs).astype(np.float32)

        # Ensure the data is in NHWC format if required
        batch_input = batch_input.reshape((-1, img_size, img_size, 3))

        total_pieces = batch_input.shape[0]
        predictions = []

        for i in range(0, total_pieces, batch_size):
            batch_data = batch_input[i:i + batch_size]
            current_batch_size = batch_data.shape[0]
            input_shape = (current_batch_size, img_size, img_size, 3)

            # Set binding shape dynamically using tensor name
            context.set_input_shape(input_tensor_name, input_shape)

            # Reallocate buffers if necessary
            input_nbytes = batch_data.nbytes
            if inputs[0].host.nbytes < input_nbytes:
                inputs[0].host = cuda.pagelocked_empty(batch_data.size, dtype=np.float32)
                inputs[0].device.free()
                inputs[0].device = cuda.mem_alloc(input_nbytes)
                bindings[0] = int(inputs[0].device)

            # Copy input data
            np.copyto(inputs[0].host, batch_data.ravel())

            # Run inference
            trt_outputs = run_inference(context, bindings, inputs, outputs, stream, current_batch_size)

            # Retrieve and reshape outputs
            output_data = outputs[0].host.copy()
            num_classes = output_data.size // current_batch_size
            batch_predictions = output_data.reshape(current_batch_size, num_classes)
            predictions.extend(batch_predictions.tolist())

        return predictions

    return predict_board(
        path,
        a1_pos,
        obtain_piece_probs_for_all_64_squares,
        previous_fen=previous_fen,
    )


def predict_board(
        board_path: str,
        a1_pos: str,
        obtain_piece_probs_for_all_64_squares,
        board_corners: (list[list[int]] | None) = None,
        previous_fen: (str | None) = None,
) -> tuple[str, list[list[int]]]:
    # Detect the board and get the corrected board image
    board_corners, corrected_board_image = detect_input_board(board_path, board_corners)

    # Split the board image into 64 squares in memory
    pieces = split_board_image_in_memory(corrected_board_image)

    # Obtain predictions for all squares in a batch
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


def detect_input_board(board_path: str, board_corners: (list[list[int]] | None) = None) -> tuple[Any, Any]:
    input_image = cv2.imread(board_path)

    # Perform board detection and correction
    image_object = detect(input_image, "images/tmp/output_board.jpg", board_corners)
    board_corners, corrected_board_image = compute_corners(image_object)

    return board_corners, corrected_board_image


def split_board_image_in_memory(board_image: np.ndarray) -> List[np.ndarray]:
    # Assumes board_image is already the corrected and cropped board image
    squares = []
    board_size = board_image.shape[0]  # Assuming square image
    square_size = board_size // 8

    for row in range(8):
        for col in range(8):
            y_start = row * square_size
            y_end = y_start + square_size
            x_start = col * square_size
            x_end = x_start + square_size
            square = board_image[y_start:y_end, x_start:x_end]
            squares.append(square)
    return squares


def check_validity_of_fen(fen: str) -> bool:
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return False

        # Count pieces using board.piece_map()
        piece_counts = {}
        for piece in board.piece_map().values():
            piece_symbol = piece.symbol()
            piece_counts[piece_symbol] = piece_counts.get(piece_symbol, 0) + 1

        # Define maximum allowed counts for each piece type
        max_counts = {
            'P': 8, 'p': 8,
            'N': 2, 'n': 2,
            'B': 2, 'b': 2,
            'R': 2, 'r': 2,
            'Q': 1, 'q': 1,
            'K': 1, 'k': 1
        }

        # Check counts against maximums
        for piece_symbol, count in piece_counts.items():
            if count > max_counts.get(piece_symbol, 0):
                return False

        return True
    except ValueError:
        return False
