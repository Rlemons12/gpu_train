import torch
import os

rank = int(os.environ.get("RANK", -1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))

torch.cuda.set_device(local_rank)

print(
    f"Rank {rank} | Local rank {local_rank} | "
    f"Device {torch.cuda.current_device()} | "
    f"{torch.cuda.get_device_name()}"
)
