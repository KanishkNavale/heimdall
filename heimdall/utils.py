import time
from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np
import torch

from heimdall.logger import OverWatch


def timeit(
    arg: Optional[Callable] = None,
    title: Optional[str] = None,
):
    logger = OverWatch("timeit")

    title = "Code block" if not isinstance(title, str) else title

    # Note: Add this function decorator at the last of all the function decorators.
    if callable(arg):

        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = arg(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"{arg.__name__} function took {duration:.4f}s to run.")
            return result

        return wrapper

    else:

        @contextmanager
        def context_wrapper():
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"{title} took {duration:.4f}s to run.")

        return context_wrapper()


def convert_numpy_to_tensor(
    func: Callable, torch_dtype: Optional[torch.dtype] = torch.float32
):
    def wrapper(*args, **kwargs):
        new_args = []
        numpy_found: bool = False

        for arg in args:
            if isinstance(arg, np.ndarray):
                numpy_found = True
                new_args.append(torch.as_tensor(arg, dtype=torch_dtype))
            else:
                new_args.append(arg)

        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                numpy_found = True
                kwargs[key] = torch.as_tensor(value, dtype=torch_dtype)

        kwargs["numpy_found"] = numpy_found

        return func(*new_args, **kwargs)

    return wrapper


def convert_tensor_to_numpy(func: Callable):
    @torch.no_grad()
    def wrapper(*args, **kwargs):
        new_args = []
        tensor_found: bool = False

        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_found = True
                new_args.append(arg.cpu().numpy())

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensor_found = True
                kwargs[key] = value.cpu().numpy()

        kwargs["tensor_found"] = tensor_found

        return func(*args, **kwargs)

    return wrapper
