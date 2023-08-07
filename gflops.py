import torch
import re
import os
from lib.model_test import D2Net
from ptflops import get_model_complexity_info

os.environ['CUDA_VISIBLE_DEVICES']='0' # set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
# Creating CNN model
model = D2Net(
    model_file='models/d2_ots.pth',
    use_relu='store_false',
    use_cuda=use_cuda
)

# Model thats already available
net = model
macs, params = get_model_complexity_info(net, (3,480,640), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)
# compute gflops
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))