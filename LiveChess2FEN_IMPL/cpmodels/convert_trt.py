import tensorrt as trt

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    # Create a TensorRT builder and network
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Create builder configuration
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision (optional)

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Get input tensor details
        input_tensor = network.get_input(0)
        print(f"Input tensor name: {input_tensor.name}")
        print(f"Input tensor shape: {input_tensor.shape}")

        # Create an optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, 224, 224, 3),   # Min shape
            (8, 224, 224, 3),   # Opt shape
            (16, 224, 224, 3)   # Max shape
        )
        config.add_optimization_profile(profile)

        # Build the serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("ERROR: Failed to build the TensorRT engine.")
            return None

        # Save the serialized engine to a file
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"TensorRT engine saved to {engine_file_path}")

# Convert model
onnx_model_path = "models/EfficientNetB7.onnx"
engine_file_path = "models/EfficientNetB7.trt"
build_engine(onnx_model_path, engine_file_path)
