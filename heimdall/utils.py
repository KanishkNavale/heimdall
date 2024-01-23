import time
from contextlib import contextmanager
from typing import Callable, Optional

from heimdall.logger import OverWatch


def timeit(arg: Optional[Callable] = None):
    logger = OverWatch("timeit")

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
            logger.info(f"Code block took {duration:.4f}s to run.")

        return context_wrapper()
