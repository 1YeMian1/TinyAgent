# ============================================================
# 环境验证
# ============================================================

import torch
import transformers
import datasets

print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Datasets:      {datasets.__version__}")
print(f"CUDA 可用:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:           {torch.cuda.get_device_name(0)}")
    print(f"显存:          {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")