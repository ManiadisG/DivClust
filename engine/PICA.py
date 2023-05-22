import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures import backbones
from architectures.layers import MultiheadLinear
from engine.criterion import PICALoss, DivClustLoss

class PICA(nn.Module):
    def __init__(self, args):
        super(PICA, self).__init__()
        self.args = args
        self.backbone = backbones.__dict__[args.backbone](args)
        self.clu_mlp = MultiheadLinear(self.backbone.output_shape, self.backbone.clusters, self.args.clusterings, True)

        self.PICALoss = PICALoss()
        self.DivLoss = DivClustLoss(**args.__dict__)

        self.current_step = 0

    def forward(self, x1, x2):

        f1, f2 = self.backbone(x1), self.backbone(x2)
        p1, p2 = F.softmax(self.clu_mlp(f1), dim=-1), F.softmax(self.clu_mlp(f2), dim=-1)
        loss_pica = self.PICALoss(p1, p2)

        diversity_loss, threshold, _ = self.DivLoss(torch.cat([p1, p2], dim=1), self.current_step)
        loss_ce_sum = sum(loss_pica) / self.args.clusterings
        diversity_loss = diversity_loss / self.args.clusterings
        loss = loss_ce_sum + diversity_loss
        if diversity_loss != 0.:
                loss = loss + diversity_loss
        self.current_step+=1
        return loss, {"loss_pica": loss_ce_sum, "loss_div": diversity_loss, "threshold": threshold}

    @torch.no_grad()
    def predict(self, x, softmax=True, return_features=False):
        f = self.backbone(x)
        p = self.clu_mlp(f)
        if softmax:
            p = F.softmax(p, dim=-1)
        if return_features:
            return p, f
        else:
            return p
