import numpy as np
import pandas as pd
import matplotlib
import sklearn
import torch

print("numpy 版本:", np.__version__)
print("pandas 版本:", pd.__version__)
print("matplotlib 版本:", matplotlib.__version__)
print("scikit-learn 版本:", sklearn.__version__)
print("PyTorch 版本:", torch.__version__)

print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("当前未检测到可用 GPU")

# 简单张量运算
x = torch.rand(3, 3)
y = torch.ones(3, 3)
print("x =")
print(x)
print("y =")
print(y)
print("x + y =")
print(x + y)