# create rknn model for add operation
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Add operation between x and y
        return torch.bitwise_xor(x, y)
        
        
x = torch.full((4096, 4096), True)
y = torch.full((4096, 4096), True)


m = Model()

torch.onnx.export(m, (x, y), "andor.onnx")


## generated onnx model, convery to rknn

from rknn.api import RKNN
rknn = RKNN()

rknn.config( target_platform='rk3588')
rknn.load_onnx(model='andor.onnx')

ret = rknn.build(do_quantization=False, dataset=None)
ret = rknn.export_rknn('andor.rknn')