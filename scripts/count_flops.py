from backbone.swapnet_v2 import swap_net_v2
from backbone.MobileNet_V2 import MobileNetV2
from thop import profile
import torch
# model=swap_net_v2(scale=1, n=[3, 8, 3])
model=MobileNetV2()
input = torch.randn(1, 3, 224, 224)
flops,params=profile(model,inputs=(input,))
print(flops,params)