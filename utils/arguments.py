import argparse
import yaml
import os
import collections
from utils.logger import logger_argparser
from utils.ddp_utils import device_argparser
from utils.misc import str2bool
import wandb
import random

def load_arguments(yaml_path):
    args_dict = config_dict_from_yaml(yaml_path)
    model_argparser = argparse_from_dict(args_dict)
    parser = argparse.ArgumentParser('Argument parser', parents=[model_argparser, logger_argparser(args_dict), device_argparser(args_dict)], conflict_handler='resolve')
    args = parser.parse_known_args()[0]
    args = parse_complex_arg(args)
    args = get_exp_dir(args)
    return args



def config_dict_from_yaml(yaml_path):
    """
    Assumes that the existence of <yaml_path>/main_config.yaml
    If a preset argument is provided, it also reads from <yaml_path>/<preset>
    """
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument("--preset", default=None, type=str, help='Used to load complementary yaml')
    parser.add_argument("--complex_arg", default=None, type=str, help='Used to modify existing arguments. Format must be <arg1_name>__<arg1_value>__<arg2_name>__<arg2_value>...')
    parser.add_argument("--bash_command", default=None, type=str, help='Used to input the command that launched the job. Read from bash script as "V="$0 $@""')
    args = parser.parse_known_args()[0]

    main_config_path = f"{yaml_path}/main_config.yaml"
    args_dict = read_yaml(main_config_path)

    if args.preset is not None:
        if ".yaml" not in args.preset:
            args.preset+=".yaml"
        preset_config_path = f"{yaml_path}/{args.preset}"
        if os.path.exists(preset_config_path):
            args_dict = read_yaml(preset_config_path, args_dict)

    args_dict['preset']=args.preset
    args_dict['complex_arg']=args.complex_arg
    args_dict['bash_command']=args.bash_command

    return args_dict



def read_yaml(path=None, parent_args=None):
    if path is None:
        return {}
    try:
        if "yaml" not in path:
            path += ".yaml"
        args = yaml.full_load(open(path, 'r'))
    except Exception as e:
        print(f"Error in reading yaml file: {e}")
        args = {}
    keys = args.keys()

    for k in keys:
        if args[k] == "None":
            args[k] = None

    if "tags" in keys:
        if args["tags"] is None:
            args["tags"] = []
        if isinstance(args["tags"], str):
            args["tags"] = args["tags"].split(' ')

    if parent_args is None:
        return args
    else:
        assert isinstance(parent_args, dict)
        if 'tags' in parent_args.keys() and 'tags' in args.keys():
            args['tags'] = list(args['tags']) + list(parent_args['tags'])
        parent_args.update(args)
        return parent_args

def parse_complex_arg(args):
    if hasattr(args, "complex_arg") and args.complex_arg is not None:
        complex_arguments = args.complex_arg.split("__")
        if len(complex_arguments) > 1:
            for i in range(0, len(complex_arguments), 2):
                if hasattr(args, complex_arguments[i]):
                    v = process_arg(complex_arguments[i+1])
                    previous_arg = getattr(args, complex_arguments[i])
                    if previous_arg is not None and type(v)!=type(previous_arg):
                        print(f"WARNING: Complex arg {complex_arguments[i]} was assigned type {type(v)}, which does not match default type {type(previous_arg)}.")
                    setattr(args, complex_arguments[i], v)
                    print(f"Overwritting argument {complex_arguments[i]} to {v}")
                else:
                    print(f"WARNING: complex_arg entry {complex_arguments[i]} was not matched to argument entry.")
    return args


def argparse_from_dict(args_dict: dict):
    parser = argparse.ArgumentParser(conflict_handler='resolve', add_help=False)
    parser.add_argument(f"--f", default=None, help="To allow notebook execution")
    parser.add_argument(f"--fff", default=None, help="To allow notebook execution")
    for k, v in args_dict.items():
        parser.add_argument(f"--{k}", default=v, type=process_arg)
    return parser


def process_arg(v):
    if isinstance(v, list) or isinstance(v, tuple):
        return [process_arg(v_) for v_ in v]
    elif isinstance(v, bool):
        return v
    elif v is None or (isinstance(v, str) and v.lower() == "none"):
        return None
    elif isinstance(v, str):
        if v.lower() in ['yes', 'true', 't', 'y', 'no', 'false', 'f', 'n']:
            return str2bool(v)
        elif v[0]=="[" and v[-1]=="]":
            v = [process_arg(v_) for v_ in v[1:-1].split(",")]
        try:
            vf = float(v)
            if '.' in v:
                return vf
            else:
                return int(vf)
        except:
            return v
    else:
        return v



def get_exp_dir(args, path=None):
    """
    Used to define the local and cloud directories for each experiment.
    It creates the local project directory if it does not exist.
    The function checks if a folder exists for that experiment and, if not, it creates it.
    It then checks to find if previous runs exist with the same ID.
    Depending on this, it assigns a path to the current experiment run (e.g. path/exp_X+1).
    :param path: The path to the directory where runs of the experiment in question are stored.
    :return: The path to the directory where the results of the current run will be stored.
    """
    args.project_name = args.project_name or "default"
    if args.output_dir is None:
        args.output_dir = './experiments/' + args.project_name
    path = path or args.output_dir

    if args.entity is None and args.wandb_mode == "online":
        args.entity = wandb.api.default_entity
    else:
        args.entity = "default"

    if not os.path.exists(path):
        used_ids = []
    else:
        used_ids = [f for f in os.listdir(path)]
    if args.wandb_mode=="online" and args.entity is not None:
        used_ids += [cr['run_id']
                     for cr in get_cloud_runs(args.project_name, args.entity)]

    if args.run_id is not None:
        error_msg = f"\nRun ID {args.run_id} not found in existing runs. If run_id is set it must have been used previously so that the run can be resumed.\n"
        assert args.run_id in used_ids, error_msg

    if args.run_name is None or args.run_name.lower()=="none":
        args.run_name = "exp"
    args.output_dir = path + "/" + args.run_name
    return args


def get_cloud_runs(project=None, entity=None):
    api = wandb.Api()
    if project is None:
        projects = [r.path for r in api.projects(entity=entity)]
    else:
        projects = [r.path for r in api.projects(
            entity=entity) if r.path[1] == project]
    runs = []
    for pr in projects:
        for ru in api.runs(pr[0] + '/' + pr[1]):
            runs.append(
                {'entity': ru.path[0], 'project': ru.path[1], 'run_id': ru.path[2], 'run_name': ru.name})
    wandb.finish()
    return runs
