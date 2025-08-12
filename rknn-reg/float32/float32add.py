# create rknn model for float32 add operation
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Simple addition operation between x and y (float32)
        return x + y
        
        
# Create float32 tensors for testing
x = torch.full((1, 10), 5.5, dtype=torch.float16)
y = torch.full((1, 10), 3.2, dtype=torch.int32)

print(f"Input x: {x}")
print(f"Input y: {y}")
print(f"Expected output: {x + y}")

m = Model()

# Export to ONNX with proper float32 support
torch.onnx.export(m, (x, y), "float32add.onnx", 
                  opset_version=11,
                  input_names=['input_x', 'input_y'],
                  output_names=['output'])

## Convert generated ONNX model to RKNN
from rknn.api import RKNN
rknn = RKNN()

rknn.config(target_platform='rk3588')
rknn.load_onnx(model='float32add.onnx')

ret = rknn.build(do_quantization=False, dataset=None)
ret = rknn.export_rknn('float32add.rknn')

print("RKNN model exported successfully!")