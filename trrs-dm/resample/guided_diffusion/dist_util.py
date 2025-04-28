"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist
from logger import logger

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    # backend = "gloo"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    
    # logger.info("Rank of current process: {}".format(MPI.COMM_WORLD.Get_rank()))
    # logger.info("Size of world: {}".format(MPI.COMM_WORLD.Get_size()))
    # logger.info("Backend:{}".format(backend))
    # logger.info("gpu:RANK:{},device_count:{}".format(os.environ["RANK"] , th.cuda.device_count()))
    # logger.info("MASTER_ADDR:{}".format(os.environ["MASTER_ADDR"]))
    # logger.info("RANK:{}".format(os.environ["RANK"]))
    # logger.info("WORLD_SIZE:{}".format(os.environ["WORLD_SIZE"]))
    # logger.info("MASTER_PORT:{}".format(os.environ["MASTER_PORT"]))
    dist.init_process_group(backend=backend, init_method="env://")
    
    dist.barrier()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
"""

python scripts/image_train.py --data_dir data/DATASET_NAME --attention_resolutions 16 
--class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True 
--noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1
--resblock_updown True --use_fp16 False --use_scale_shift_norm True 
--lr 2e-5 --batch_size 8 --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir logs


python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpu_using_launch.py


python -m torch.distributed.launch --nproc_per_node=2 --use_env image_train.py --data_dir data/DATASET_NAME --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir logs


torchrun --nproc_per_node=2 train.py --use_env image_train.py --data_dir data/celeba_hq_256 --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir logs

"""