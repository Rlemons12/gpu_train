import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear


def worker(rank: int, world_size: int, store_path: str) -> None:
    torch.cuda.set_device(rank)
    store = dist.FileStore(store_path, world_size)

    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    model = Linear(1024, 1024).cuda()
    model = FSDP(model)

    x = torch.randn(8, 1024, device="cuda")
    y = model(x)

    if dist.get_rank() == 0:
        print("FSDP forward pass OK", flush=True)

    dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    world_size = torch.cuda.device_count()
    run_dir = Path(__file__).resolve().parent
    store_path = str(run_dir / "filestore_rdzv_fsdp")

    try:
        os.remove(store_path)
    except FileNotFoundError:
        pass

    mp.spawn(worker, args=(world_size, store_path), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
