import onnx

# Load the ONNX model
onnx_model = onnx.load("../data/models/ResNet152V3.onnx")

print("Operators in the ONNX model:")
for node in onnx_model.graph.node:
    print(f"{node.op_type}")

# Modify the input tensor to set batch dimension to 'None'
input_tensor = onnx_model.graph.input[0]  # Assuming single input
input_tensor.type.tensor_type.shape.dim[0].dim_param = 'None'

# Save the modified ONNX model
onnx.save(onnx_model, "../data/models/ResNet152V3_fixed.onnx")
print("ONNX model with fixed batch dimension saved!")

