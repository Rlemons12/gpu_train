import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear


def setup_distributed():
    """
    Safe distributed init:
    - Works with torchrun
    - Works with single-GPU (WORLD_SIZE=1)
    """

    if dist.is_initialized():
        return

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )


def main():
    # --------------------------------------------------
    # Distributed setup
    # --------------------------------------------------
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[INIT] world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"[GPU] {torch.cuda.get_device_name(local_rank)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = Linear(1024, 1024).cuda()

    fsdp_model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
    )

    # --------------------------------------------------
    # Forward + backward sanity
    # --------------------------------------------------
    x = torch.randn(8, 1024, device="cuda")
    y = fsdp_model(x).sum()
    y.backward()

    if rank == 0:
        print("✅ FSDP forward + backward pass OK")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
