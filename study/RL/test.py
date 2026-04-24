import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
print(f"MPS 是否已构建: {torch.backends.mps.is_built()}")