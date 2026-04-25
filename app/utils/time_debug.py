import time
import os
from functools import wraps

DEBUG_TIMING = os.environ.get('FLASK_DEBUG', '0') == '1'

def timed(fn):
    if not DEBUG_TIMING:
        return fn  # zero overhead in production
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"[TIMER] {fn.__qualname__}: {time.perf_counter() - start:.3f}s")
        return result
    return wrapper