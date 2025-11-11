from typing import Tuple, Union, Dict, Any

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .logger import LOGGER

def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK", "") != "":
        local_rank = int(os.environ["LOCAL_RANK"])
    elif os.environ.get("SLURM_LOCALID", "") != "":
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_rank = -1
    return local_rank

def _parse_slurm_nlist(val: str) -> int:
    """
    解析 SLURM 可能的格式: '4', '1,1,2', '4(x2)', '2(x3),1' 等
    返回总数或第一个数字（这里我们需要“每节点 GPU 数”，尽量取第一段的整数）
    """
    if not val:
        return 0
    val = val.strip()
    # 形如 '4(x2)' 或 '4' -> 取4
    m = re.match(r"(\d+)(?:\(x\d+\))?$", val)
    if m:
        return int(m.group(1))
    # 形如 '1,1,2' -> 取第一个
    parts = re.split(r"[,\s]+", val)
    for p in parts:
        p = p.strip()
        if p.isdigit():
            return int(p)
    # 兜底尝试提取第一个数字
    m2 = re.search(r"\d+", val)
    return int(m2.group(0)) if m2 else 0

def load_init_param(opts):
    """
    Load parameters for the rendezvous distributed procedure
    优先使用 torchrun 注入的环境变量；仅在没有 WORLD_SIZE/RANK 的情况下再从 SLURM 推断
    """
    # GPUs per node（仅作为 fallback 计算 rank 用）
    if os.environ.get("SLURM_NTASKS_PER_NODE", ""):
        num_gpus = _parse_slurm_nlist(os.environ["SLURM_NTASKS_PER_NODE"])
    elif os.environ.get("SLURM_TASKS_PER_NODE", ""):
        num_gpus = _parse_slurm_nlist(os.environ["SLURM_TASKS_PER_NODE"])
    else:
        # 受 CUDA_VISIBLE_DEVICES 影响
        num_gpus = torch.cuda.device_count()

    # world size（优先用 torchrun 的 WORLD_SIZE）
    if os.environ.get("WORLD_SIZE", ""):
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.environ.get("SLURM_JOB_NUM_NODES", ""):
        num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        world_size = num_nodes * num_gpus
    elif os.environ.get("SLURM_NNODES", ""):
        num_nodes = int(os.environ["SLURM_NNODES"])
        world_size = num_nodes * num_gpus
    else:
        raise RuntimeError("Can't find any WORLD_SIZE; please launch with torchrun or set SLURM_* properly.")
    opts.world_size = world_size

    # rank（优先用 torchrun 的 RANK）
    if os.environ.get("RANK", ""):
        rank = int(os.environ["RANK"])
    elif os.environ.get("SLURM_PROCID", ""):
        rank = int(os.environ["SLURM_PROCID"])
    else:
        # fallback: 需要 node_rank 与 local_rank
        if os.environ.get("NODE_RANK", ""):
            opts.node_rank = int(os.environ["NODE_RANK"])
        elif os.environ.get("SLURM_NODEID", ""):
            opts.node_rank = int(os.environ["SLURM_NODEID"])
        else:
            raise RuntimeError("Can't find any RANK or NODE_RANK for fallback.")
        if not hasattr(opts, "local_rank") or opts.local_rank is None or opts.local_rank < 0:
            # 从环境再兜底取
            opts.local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if opts.local_rank < 0:
                raise RuntimeError("local_rank is required in fallback path.")
        rank = opts.local_rank + opts.node_rank * max(1, num_gpus)
    opts.rank = rank

    # init method（env:// 需要 MASTER_ADDR/MASTER_PORT 已设置；torchrun 会注入）
    init_method = "env://"

    # backend（CUDA 下推荐 nccl）
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    return {
        "backend": backend,
        "init_method": init_method,
        "rank": opts.rank,
        "world_size": opts.world_size,
    }

def init_distributed(opts):
    # ✅ 防止重复初始化
    if dist.is_initialized():
        return

    init_param = load_init_param(opts)

    # 可选：提前检查 master 地址/端口（在非 torchrun 情况下）
    if init_param["init_method"] == "env://":
        if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
            raise RuntimeError("env:// requires MASTER_ADDR and MASTER_PORT to be set.")

    print(f"[DDP] Init rank={init_param['rank']} world_size={init_param['world_size']} backend={init_param['backend']}")
    dist.init_process_group(**init_param)
    # 可选：同步一次
    dist.barrier()

def set_cuda(opts) -> Tuple[bool, int, torch.device]:
    """
    Initialize CUDA for distributed computing
    """
    local_rank = get_local_rank()
    opts.local_rank = local_rank
    
    print("local_rank: ", local_rank)

    if not torch.cuda.is_available():
        assert local_rank == -1, local_rank
        return True, 0, torch.device("cpu")

    # 分布式多卡
    if opts.local_rank != -1:
        init_distributed(opts) #分布式环境初始化
        torch.cuda.set_device(opts.local_rank) #绑定当前进程的GPU
        device = torch.device("cuda", opts.local_rank)
        n_gpu = 1
        default_gpu = dist.get_rank() == 0
        if default_gpu:
            LOGGER.info(f"Found {dist.get_world_size()} GPUs") #主进程
    else:
        default_gpu = True
        device = torch.device("cuda", opts.cuda_device)
        print(f"use device {device}")
        n_gpu = torch.cuda.device_count()

    return default_gpu, n_gpu, device


def wrap_model(
    model: torch.nn.Module, 
    device: torch.device, 
    local_rank: int,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    model.to(device)

    if local_rank != -1:
        model = DDP(
            model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters
        )
        # At the time of DDP wrapping, parameters and buffers (i.e., model.state_dict()) 
        # on rank0 are broadcasted to all other ranks.
    
    # a single card is enough for our model
    # elif torch.cuda.device_count() > 1:
    #     LOGGER.info("Using data parallel")
    #     model = torch.nn.DataParallel(model)

    return model