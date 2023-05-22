import sys
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def export_fn(fn):
    """
    Implementation adapted from https://stackoverflow.com/a/41895257
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        name = fn.__name__
        all_ = mod.__all__
        assert name not in mod.__all__
        all_.append(name)
    else:
        mod.__all__ = [fn.__name__]
    return fn

def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)