import tensorrt as trt

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Configure the builder settings
        config = builder.create_builder_config()

        # Enable FP16 mode if the platform supports it
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse the ONNX model
        with open(onnx_file_path, "rb") as model:
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

        # Build the engine
        engine = builder.build_engine(network, config)

        # Serialize the engine to a file
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

        return engine

onnx_path = "../data/models/ResNet152V2_103.onnx"
trt_path = "../data/models/ResNet152V2_103.trt"
build_engine(onnx_path, trt_path)

