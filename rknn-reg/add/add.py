# create rknn model for add operation
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Add operation between x and y
        return x + y
        
        
x = torch.full((1, 1024), 1, dtype=torch.float16)
y = torch.full((1, 1024), 1, dtype=torch.float16)


m = Model()

torch.onnx.export(m, (x, y), "add.onnx", opset_version=12)


## generated onnx model, convery to rknn

from rknn.api import RKNN
rknn = RKNN()

rknn.config( target_platform='rk3588' )
rknn.load_onnx(model='add.onnx')

ret = rknn.build(do_quantization=False, dataset=None)
ret = rknn.export_rknn('add.rknn')