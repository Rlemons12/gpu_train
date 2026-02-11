import torch

def verify_gpu_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    prop = torch.cuda.get_device_properties(0)
    cc = f"sm_{prop.major}{prop.minor}"

    if prop.major < 12:
        raise RuntimeError(
            f"Unsupported GPU architecture {cc}. "
            "RTX 5090 requires sm_120 + PyTorch nightly (cu128)."
        )
v