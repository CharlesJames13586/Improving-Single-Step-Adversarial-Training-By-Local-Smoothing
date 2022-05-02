import torch
import torch.distributed as dist


def reduce_value(value, average=False):
    """假设调用者只在多GPU下调用，不做其判断"""
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
