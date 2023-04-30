from utils.logger import Logger
from utils.misc import export_fn
import torch
from torch.cuda.amp import autocast, GradScaler
import os
from engine.criterion import clustering_accuracy_metrics

@export_fn
class Trainer:
    def __init__(self, model, optimizer, args, logger:Logger):
        self.train_step=0
        self.epoch=0
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.device = args.gpu

        self.scaler = GradScaler()
        self.mixed_precision = args.__dict__.get("mixed_precision", True)
        self.logger.print(f"Mixed precision: {'ON' if self.mixed_precision else 'OFF'}")

    def train_epoch(self, train_dataloader, eval_dataloader, print_interval=25, eval=True):
        device = self.args.gpu
        self.model.train()

        epoch_steps = len(train_dataloader)
        for batch_id, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()

            idx, samples, annotations = batch
            samples = samples.permute(1,0,2,3,4).to(device,non_blocking=True)
            annotations = annotations.cuda(device,non_blocking=True)

            with autocast(self.mixed_precision):
                loss, metrics_dict = self.model(samples[0], samples[1])

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics_dict.update({"loss":loss})
            self.logger.log(metrics_dict)
            if print_interval>0 and batch_id%print_interval==0:
                self.logger.print_epoch_progress(batch_id, epoch_steps, self.epoch, self.args.epochs)

        if eval:
            self.logger.print(f"Evaluating")
            self.model.eval()
            cluster_labels = []
            ground_truth_labels = []
            confidence = 0
            samples = 0

            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    if step % 10 == 0:
                        self.logger.print(f"Eval. step {step} of {len(eval_dataloader)}")
                    index, x, target = batch
                    x = x.cuda(self.device)
                    preds = self.model.predict(x)
                    confidence += preds.max(-1)[0].sum(-1).mean()
                    samples += x.shape[0]
                    cluster_labels.append(torch.argmax(preds, dim=-1).data.cpu())
                    ground_truth_labels.append(target.data.cpu())

            ground_truth_labels = torch.cat(ground_truth_labels, dim=0)
            cluster_labels = torch.cat(cluster_labels, dim=1)
            if len(ground_truth_labels.shape) != 1:
                ground_truth_labels = ground_truth_labels.permute(1, 0)
            metrics_ = clustering_accuracy_metrics(cluster_labels, ground_truth_labels)
            eval_metrics = {"eval_confidence": confidence / samples}
            for k, v in metrics_.items():
                eval_metrics[f"{k}_eval"] = v
            self.logger.log(eval_metrics)

            if self.epoch == (self.args.epochs - 1):
                torch.save({"ground_truth": ground_truth_labels.cpu().numpy(), "clusters": cluster_labels.cpu().numpy()}, self.args.experiment_dir + "/outcomes")

        self.logger.epoch_end(self.epoch, self.args.epochs)
        self.epoch+=1
        
    # def get_state_dict(self):
    #     if self.args.rank==0:
    #         return {"model": self.model.state_dict(),
    #                     "criterion":self.criterion.state_dict(),
    #                     "optimizer":self.optimizer.state_dict(),
    #                     "scheduler": self.scheduler.state_dict(),
    #                     "args":self.args,
    #                     "train_step":self.train_step,
    #                     "epoch":self.epoch}

    # def load_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict["model"])
    #     self.criterion.load_state_dict(state_dict["criterion"])
    #     self.optimizer.load_state_dict(state_dict["optimizer"])
    #     self.scheduler.load_state_dict(state_dict["scheduler"])
    #     self.args = state_dict["args"]
    #     self.train_step = state_dict["train_step"]
    #     self.epoch = state_dict["epoch"]

    # def load_checkpoint(self, checkpoint_path=None):
    #     if checkpoint_path is None or str(checkpoint_path).lower()=="none" or str(checkpoint_path).lower()=="false":
    #         return None
    #     elif str(checkpoint_path).lower()=="true":
    #         checkpoint_path = f"{self.args.output_dir}/checkpoint.pth"
    #     if os.path.exists(checkpoint_path):
    #         return torch.load(checkpoint_path), checkpoint_path
    #     else:
    #         return None