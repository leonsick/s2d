# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from contextlib import contextmanager
from functools import wraps
import torch
from torch.cuda.amp import autocast

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.
    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.
    Args:
        func: a stateless callable that takes tensor-like objects as arguments
    Returns:
        a callable which retries `func` if OOM is encountered.
    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU
    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.
        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu").to(torch.float32)
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs to CPU due to CUDA OOM")
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        with autocast(enabled=False):
            return func(*new_args, **new_kwargs)

    return wrapped

def model_retry_if_cuda_oom(model: torch.nn.Module):
    """
    Creates a wrapper around an nn.Module's forward call to retry on PyTorch's
    CUDA OOM error. It will first retry after calling `torch.cuda.empty_cache()`.
    If that still fails, it will attempt to move the model and inputs to CPU,
    run the operation, and then move the model back to the GPU.

    Args:
        model: The nn.Module instance to be wrapped.

    Returns:
        A callable that retries the model's forward call on an OOM error.

    Note:
        The return values from the CPU execution will be on the CPU.
    """

    def _recursive_to_cpu(x):
        if isinstance(x, torch.Tensor):
            try:
                if x.device.type == "cuda":
                    return x.to("cpu")
            except AttributeError:
                pass
            return x
        elif isinstance(x, (list, tuple)):
            return type(x)(_recursive_to_cpu(item) for item in x)
        elif isinstance(x, dict):
            return {k: _recursive_to_cpu(v) for k, v in x.items()}
        else:
            return x

    @wraps(model.forward)
    def wrapped(*args, **kwargs):
        # First attempt on GPU
        with _ignore_torch_cuda_oom():
            return model.forward(*args, **kwargs)

        # Clear cache and retry on GPU
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return model.forward(*args, **kwargs)

        # Try on CPU.
        logger = logging.getLogger(__name__)
        logger.info(f"Attempting to run model {type(model).__name__} on CPU due to CUDA OOM")

        original_device = next(model.parameters()).device
        try:
            model.to("cpu")
            new_args = _recursive_to_cpu(args)
            new_kwargs = _recursive_to_cpu(kwargs)
            with autocast(enabled=False):
                return model.forward(*new_args, **new_kwargs)
        finally:
            # Ensure the model is always moved back to its original device
            model.to(original_device)

    return wrapped