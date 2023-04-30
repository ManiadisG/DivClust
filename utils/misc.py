import sys
import math
import time

def split_to_intervals(end_value, intervals, start_value=0):
    chunk_size = (end_value - start_value) / intervals
    intervals = [math.floor(start_value + ch * chunk_size) for ch in range(intervals)]
    intervals.append(end_value)
    return intervals

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

class Timer:
    def __init__(self):
        self.times_tic = {}
        self.times_toc = {}

    def tic(self, names=None):
        if names is None:
            names=["time"]
        elif isinstance(names, str):
            names = [names]
        for name in names:
            self.times_tic[name]=time.time()

    def toc(self, names=None, reset=False):
        if names is None:
            names=["time"]
        elif isinstance(names, str):
            names = [names]
        for name in names:
            if name in self.times_tic.keys() and self.times_tic[name] is not None:
                toc=time.time()-self.times_tic[name]
                if reset or name not in self.times_toc.keys():
                    self.times_toc[name]=toc
                else:
                    self.times_toc[name]+=toc
       
    def reset(self):
        self.times_tic = {}
        self.times_toc = {}

    def get_time(self, name=None):
        if name is None:
            return self.times_toc
        else:
            assert name in self.times_toc.keys()
            return self.times_toc[name]


def is_list_or_tuple(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)