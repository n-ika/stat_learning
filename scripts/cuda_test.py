
import sys, traceback
print("Python:", sys.version)
print("Executable:", sys.executable)
try:
    import torch
    print("torch file:", torch.__file__)
    print("torch version:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda is_available:", torch.cuda.is_available())
except Exception as e:
    print("ERROR during import torch:", repr(e))
    traceback.print_exc()

import torch

print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())