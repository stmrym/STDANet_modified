from functools import wraps
import time

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print("{} s in {}".format(elapsed_time, func.__name__))
        return result
    return wrapper