import onnx

onnx_model_fixed = onnx.load("../data/models/ResNet152V3_fixed.onnx")

for input_tensor in onnx_model_fixed.graph.input:
    dims = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input: {input_tensor.name}, Shape: {dims}")

