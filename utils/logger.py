
import time
import warnings
import torch
import torch.distributed as dist
import wandb
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import sys
import logging
import functools
from termcolor import colored
import argparse
import traceback

os.environ['WANDB_START_METHOD'] = "thread"
os.environ['WANDB_SILENT'] = "true"
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def logger_argparser(args_dict=None):
    parser = argparse.ArgumentParser(conflict_handler='resolve',add_help=False)
    parser.add_argument(
        "--output_dir", default=args_dict.get("output_dir", None), type=str, help="Experiment parent directory")
    parser.add_argument(
        "--entity", default=args_dict.get("entity", None), type=str, help="Wandb entity")
    parser.add_argument(
        "--project_name", default=args_dict.get("project_name", None), type=str, help="Wandb project name")
    parser.add_argument(
        "--run_name", default=args_dict.get("run_name", None), type=str, help="Name of current experiment")
    parser.add_argument(
        "--run_id", default=args_dict.get("run_id", None), type=str, help="ID of current experiment")
    parser.add_argument(
        "--group", default=args_dict.get("group", None), type=str, help="Wandb group")
    parser.add_argument(
        "--tags", default=args_dict.get("tags", None), type=str, help="Wandb tags")
    parser.add_argument(
        "--notes", default=args_dict.get("notes", None), type=str, help="Wandb notes")
    parser.add_argument(
        "--wandb_mode", default=args_dict.get("wandb_mode", "off"), type=str, help="Wandb mode. Possible options: online/offline/off")
    parser.add_argument(
        "--resume", default=args_dict.get("resume"), type=bool, help="Whether to resume run")
    return parser


class Logger:
    def __init__(self, args):

        self.output_dir = args.output_dir
        self.entity = args.entity
        self.project_name = args.project_name or "ML"
        self.run_name = args.run_name or "exp"
        self.group = args.group
        self.tags = args.tags
        self.notes = args.notes
        self.wandb_mode = args.wandb_mode
        self.resume = args.resume
        self.args = args

        self.config_dict = args.__dict__
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.wandb_init()
        self.logger = create_logger(self.output_dir)
        self.print(f"Initializing experiment {self.run_name}\n")
        self.metric_handler = MetricHandler()
        self.print_args()

    def print_epoch_progress(self, step, of_steps=None, epoch=None, of_epochs=None):
        if step == 0:
            self.epoch_start_time = time.time()
        epoch_time = time.time()-self.epoch_start_time
        steps_per_s = round(step / epoch_time, 2)
        text = "id=" + self.run_name + " | "
        if epoch is not None:
            text += f"Ep. {epoch}"
            if of_epochs is not None:
                text+=f"/{of_epochs}"
            text+=" - "
        text += f"Step {step}"
        if of_steps is not None:
            text = text + f"/{of_steps}"
        text += f" ({steps_per_s} it/s) |"
        if of_steps is not None:
            prog_ratio = round(step / of_steps * 100, 1)
            text += (int(prog_ratio // 5) * "#" +
                     int(20 - prog_ratio // 5) * " " + "| - ")
        else:
            text += " "
        metrics = self.metric_handler.get_avg()
        if len(metrics.keys()) != 0:
            for k, v in metrics.items():
                text += f"{k}={self.rounding(v)} - "
            text = text[:-3]
        self.print(text)

    def print_epoch_end(self, epoch, of_epochs=None):
        """
        Prints epoch-wise progress bar with requested metrics (if they have been logged)
        """
        epoch_time = time.time()-self.epoch_start_time
        text = f"id={self.run_name} | Ep. {epoch}"
        if of_epochs is not None:
            text += f"/{of_epochs}"
        text += f" | Time: {convert_time(epoch_time)} | "
        metrics = self.metric_handler.get_avg()
        if len(metrics.keys()) != 0:
            for k, v in metrics.items():
                text += f"{k}={self.rounding(v)} - "
            text = text[:-3]
        self.print(text, end='\n')

    def epoch_end(self, epoch, of_epochs=None, reset_metrics=True):
        self.print_epoch_end(epoch, of_epochs)
        avg_metrics = self.metric_handler.get_avg()
        if self.wandb_mode!="off":
            wandb.log(avg_metrics, step=epoch)
        if reset_metrics:
            self.metric_handler.reset()
        if self.wandb_mode!="off":
            self.upload_logs()

    def rounding(self, v):
        """
        Rounds values for printing
        """
        v = float(v)
        if v<0:
            negative=True
            v = -v
        else:
            negative=False
        if v==0:
            return v
        elif v >= 100:
            v = round(v, 1)
        elif v >= 10:
            v = round(v, 2)
        elif v >= 1:
            v = round(v, 3)
        else:
            decimal_count=0
            while v*(10**decimal_count)<1:
                decimal_count+=1
            v = round(v, decimal_count+2)
        if negative:
            v = -v
        return v

    def log(self, metrics:dict):
        self.metric_handler.add_metrics(metrics)

    def log_and_write(self, metrics: dict, epoch=None):
        wandb.log(metrics, step=epoch)

    def print(self, msg, end="\n"):
        if end == "\r":
            print(self.adjust_print_string_length(msg), end="\r")
        else:
            self.logger.info(msg)

    def error(self, e):
        msg = traceback.format_exc()
        self.logger.error(msg, exc_info=True)
        self.logger.error(e, exc_info=True)

    def adjust_print_string_length(self, text):
        """
        Adjusts printed messages to fill the terminal and overwrite previous prints on the same line
        """
        try:
            terminal_len = os.get_terminal_size()[0]
        except Exception as e:
            self.print(f"Error in getting actual terminal size: {e}")
            terminal_len = 200
        if terminal_len == 0:
            terminal_len = 200
        if len(text) < terminal_len:
            text = text + (terminal_len - len(text)) * " "
        elif len(text) > terminal_len:
            text = text[:(terminal_len - 5)] + ' ...'
        return text

    def wandb_init(self):
        if self.wandb_mode=="off":
            return None
        wandb.init(project=self.project_name, entity=self.entity, config=self.config_dict, group=self.group,
                   tags=self.tags, notes=self.notes, dir=self.output_dir, id=self.args.run_id,
                   name=self.run_name, resume="allow", mode=self.wandb_mode, anonymous="allow")

    def print_args(self):
        to_write = ""
        keys = list(self.config_dict.keys())
        keys.sort()
        for k in keys:
            self.print(f"{k} = {self.config_dict[k]}")
            to_write+=f"{k} = {self.config_dict[k]}\n"
        self.print("")
        with open(f"{self.output_dir}/config.txt", "w") as f:
            f.write(to_write)

    def upload_logs(self):
        try:
            wandb.save(f"{self.output_dir}/log.txt", base_path=self.output_dir)
        except Exception as e:
            msg = traceback.format_exc()
            self.print(f"Failed to upload log")
            self.print(f"Error: {msg}")

    def finish(self, crashed=False):
        if self.wandb_mode!="off":
            self.upload_logs()
            if crashed:
                wandb.finish(quiet=True, exit_code=1)
            else:
                wandb.finish()

class MetricHandler:
    def __init__(self):
        self.metrics = {}

    def reset(self):
        self.metrics = {}

    def add_metrics(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            if k not in self.metrics.keys():
                self.metrics[k] = {"value": v, "count": torch.ones(
                    (1,), device=torch.cuda.current_device())}
            else:
                self.metrics[k]["value"] += v
                self.metrics[k]["count"] += 1

    def get_avg(self):
        metric_averages = {}
        for k, v in self.metrics.items():
            if not isinstance(v["value"], torch.Tensor):
                k_value = torch.tensor(v["value"], device=torch.cuda.current_device())
            else:
                k_value = v["value"].clone().detach()
            k_count = v["count"].clone()
            metric_averages[k] = k_value/k_count
        return metric_averages


@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addFilter(NoParsingFilter())
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(
        fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        cut_messages = []
        for cm in cut_messages:
            if cm in record.filename:
                return False
        return True

def convert_time(t):
    """
    seconds to HH:MM string format
    """
    h = int(t // 3600)
    m = int((t - h * 3600) // 60)
    s = str(h).zfill(2) + ":" + str(m).zfill(2)
    return s

