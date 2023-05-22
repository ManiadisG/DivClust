import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures import backbones
from architectures.layers import MultiheadLinear
from engine.criterion import CCLoss, DivClustLoss

class CC(nn.Module):
    def __init__(self, args):
        super(CC, self).__init__()
        self.args = args
        self.backbone = backbones.__dict__[args.backbone](args)
        self.id_mlp = nn.Sequential(nn.Linear(self.backbone.output_shape, self.backbone.output_shape), nn.ReLU(),
                                    nn.Linear(self.backbone.output_shape, self.args.proj_dim))
        self.clu_mlp = nn.Sequential(
            MultiheadLinear(self.backbone.output_shape, self.backbone.output_shape, self.args.clusterings, True), nn.ReLU(),
            MultiheadLinear(self.backbone.output_shape, self.args.clusters, self.args.clusterings, True))

        self.CCLoss = CCLoss()
        self.DivLoss = DivClustLoss(**args.__dict__)

        self.current_step = 0

    def forward(self, x1, x2):

        f1, f2 = self.backbone(x1), self.backbone(x2)
        p1, p2 = F.softmax(self.clu_mlp(f1), dim=-1), F.softmax(self.clu_mlp(f2), dim=-1)
        z1, z2 = self.id_mlp(f1), self.id_mlp(f2)

        loss_ce, loss_ne, loss_cc = self.CCLoss(p1, p2, z1, z2)
        diversity_loss, threshold, _ = self.DivLoss(torch.cat([p1, p2], dim=1), self.current_step)
        loss_ce_sum = sum(loss_ce) / self.args.clusterings
        loss_ne_sum = sum(loss_ne) / self.args.clusterings
        diversity_loss = diversity_loss / self.args.clusterings
        loss = loss_ce_sum + loss_ne_sum + loss_cc + diversity_loss
        self.current_step+=1
        return loss, {"loss_cc": loss_cc, "loss_ce": loss_ce_sum, "loss_ne": loss_ne_sum, "loss_div": diversity_loss, "threshold": threshold}

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
