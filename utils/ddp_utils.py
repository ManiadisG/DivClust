
import torch
import torch.distributed as dist
import os
import collections
import argparse


def device_argparser(args_dict=None):
    parser = argparse.ArgumentParser(conflict_handler='resolve',add_help=False)

    default_gpus = args_dict.get("gpus",None)
    if default_gpus is not None and isinstance(default_gpus, str):
        default_gpus = default_gpus.replace(" ","").split(",")
        default_gpus = [int(dg) for dg in default_gpus]
    parser.add_argument(
        "--gpus", default=default_gpus, type=int, nargs="+", help="To be used if individual gpus are to be selected")
    parser.add_argument(
        "--num_workers", default=args_dict.get("num_workers", 2), type=int, help="Num workers per dataloader")
    return parser


def cat_all_gather(tensor: torch.Tensor):
    """
    Applies all_gather AND concatenates the gathered tensors
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)
    else:
        return tensor


def init_distributed_mode(args):
    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        if hasattr(args, 'gpus') and (isinstance(args.gpus, list) or isinstance(args.gpus, tuple)):
            args.gpu = args.gpus[int(os.environ['LOCAL_RANK'])]
        else:
            args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    except Exception as e:
        print(f"WARNIGN: DDP not initialized\nError: {e}")



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def batch_size_per_device(batch_size):
    devices = int(os.environ['WORLD_SIZE'])
    if batch_size%devices!=0:
        print("WARNING: Batch size not evenly divisible with devices")
    return batch_size//devices