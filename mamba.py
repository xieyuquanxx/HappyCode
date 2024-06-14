import os

import torch

from model import Mamba, Mamba2

"""
export LC_ALL="en_US.UTF-8"
export LIBRARY_PATH="/usr/local/cuda-11.8/lib64/stubs/"
sudo ldconfig /usr/local/cuda-11.8/lib64

https://github.com/state-spaces/mamba/issues/390
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

batch, length, dim = 2, 64, 512
x = torch.randn(batch, length, dim).to("cuda:7")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=64,  # SSM state expansion factor
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
).to("cuda:7")
y = model(x)
