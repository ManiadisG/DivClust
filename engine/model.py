
import torch.nn as nn
import architectures.backbones as backbones
from utils.misc import export_fn

@export_fn
def build_model(args, logger=None):
    if args.clustering_framework.lower() == "cc":
        from engine.CC import CC
        model = CC(args)
        model_name="CC"
    if args.clustering_framework.lower() == "pica":
        from engine.PICA import PICA
        model = PICA(args)
        model_name="PICA"
    if logger is not None:
        p_count, p_count_train = 0, 0
        for p in model.parameters():
            p_count+=p.numel()
            if p.requires_grad:
                p_count_train+=p.numel()
        logger.print(f"{model_name} model built. {p_count} parameters, {p_count_train} trainable parameters.")
    return model
