import torch

def get_device():
    # 检查原生 torch.xpu
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"Native Intel XPU detected: {torch.xpu.get_device_name(0)}")
            return torch.device("xpu")
    except Exception:
        pass
    
    print("XPU not found, falling back to CPU.")
    return torch.device("cpu")

DEVICE = get_device()
XPU_AVAILABLE = (DEVICE.type == "xpu")