"""This is the main program for converting board images into FENs."""

import argparse

from keras.api.applications.imagenet_utils import (
    preprocess_input as prein_squeezenet,
)
from keras.api.applications.efficientnet_v2 import preprocess_input as prein_efficientnet
from keras.api.applications.resnet_v2 import preprocess_input as prein_resnet
from keras.api.applications.xception import preprocess_input as prein_xception

from lc2fen.predict_board import (
    predict_board_keras,
    predict_board_onnx,
    predict_board_trt,
    predict_board_with_piece_detection
)


def preprocess_input2(x):
    """Preprocess the input image according to AlexNet requirements."""
    x /= 127.5
    x -= 1.0
    return x


ACTIVATE_KERAS = False
MODEL_PATH_KERAS = "/home/adi/LiveChess2FEN/data/models/squeezenet_epoch_73.keras"
IMG_SIZE_KERAS = 227
PRE_INPUT_KERAS = prein_resnet

ACTIVATE_ONNX = False
MODEL_PATH_ONNX = "data/models/ResNet152V2.onnx"
IMG_SIZE_ONNX = 224
PRE_INPUT_ONNX = prein_resnet

ACTIVATE_TRT = False
MODEL_PATH_TRT = "data/models/ResNet152V2_103.trt"
IMG_SIZE_TRT = 224
PRE_INPUT_TRT = prein_resnet

ACTIVATE_PT = False
MODEL_PATH_PT = "data/models/best.pt"


def parse_arguments() -> tuple[str, str, str | None]:
    global ACTIVATE_KERAS, ACTIVATE_ONNX, ACTIVATE_TRT, ACTIVATE_PT

    parser = argparse.ArgumentParser(
        description="Predicts board configuration(s) (FEN string(s)) from "
                    "image(s)."
    )

    parser.add_argument(
        "path",
        help="Path to the image or folder you wish to predict the FEN(s) for",
    )
    parser.add_argument(
        "a1_pos",
        help="Location of the a1 square in the chessboard image(s) "
             "(B = bottom, T = top, R = right, L = left)",
        choices=["BL", "BR", "TL", "TR"],
    )
    parser.add_argument(
        "previous_fen",
        nargs="?",
        help="FEN string of the previous board position (if "
             "you are predicting the FEN for a single image and if "
             "the previous board position is known)",
    )

    inf_engine = parser.add_mutually_exclusive_group(required=True)
    inf_engine.add_argument(
        "-k", "--keras", help="run inference using Keras", action="store_true"
    )
    inf_engine.add_argument(
        "-o", "--onnx", help="run inference using ONNXRuntime", action="store_true",
    )
    inf_engine.add_argument(
        "-t", "--trt", help="run inference using TensorRT", action="store_true"
    )
    inf_engine.add_argument(
        "-p", "--pt", help="run inference using PyTorch", action="store_true"
    )

    args = parser.parse_args()

    if args.keras:
        ACTIVATE_KERAS = True
    elif args.onnx:
        ACTIVATE_ONNX = True
    elif args.trt:
        ACTIVATE_TRT = True
    elif args.pt:
        ACTIVATE_PT = True
    else:
        ValueError("No inference engine selected. This should be unreachable.")

    return args.path, args.a1_pos, args.previous_fen


def main():
    """Parse the arguments and print the predicted FEN."""
    path, a1_pos, previous_fen = parse_arguments()

    if ACTIVATE_KERAS:
        fen, _ = predict_board_keras(
            MODEL_PATH_KERAS,
            IMG_SIZE_KERAS,
            PRE_INPUT_KERAS,
            path,
            a1_pos,
            previous_fen=previous_fen,
        )
    elif ACTIVATE_ONNX:
        fen, _ = predict_board_onnx(
            MODEL_PATH_ONNX,
            IMG_SIZE_ONNX,
            PRE_INPUT_ONNX,
            path,
            a1_pos,
            previous_fen=previous_fen,
        )
    elif ACTIVATE_TRT:
        fen, _ = predict_board_trt(
            MODEL_PATH_TRT,
            IMG_SIZE_TRT,
            PRE_INPUT_TRT,
            path,
            a1_pos,
            previous_fen=previous_fen,
        )
    elif ACTIVATE_PT:
        fen, _ = predict_board_with_piece_detection(
            MODEL_PATH_PT,
            MODEL_PATH_KERAS,
            IMG_SIZE_KERAS,
            PRE_INPUT_KERAS,
            path,
            a1_pos,
            previous_fen=previous_fen,
        )
    else:
        fen = None
        ValueError("No inference engine selected. This should be unreachable.")

    print(fen)


if __name__ == "__main__":
    main()
