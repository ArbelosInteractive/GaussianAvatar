import ctypes

try:
    ctypes.CDLL("libcuda.so")
    print("libcuda.so loaded successfully")
except Exception as e:
    print(f"Failed to load libcuda.so: {e}")


import torch

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
print("Torch current device:", torch.cuda.current_device())
print("Torch device name:", torch.cuda.get_device_name(0))