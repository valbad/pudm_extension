"""General training and checkpoint utilities."""
import os
import re
import random
import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if summary_writer is not None:
            summary_writer.add_scalar(self.name, val, global_step=global_step)


def find_max_epoch(path, ckpt_name, mode='max', return_num_ckpts=False):
    """Find maximum epoch/iteration in path, formatted ${ckpt_name}_${n_iter}.pkl"""
    files = os.listdir(path)
    iterations = []
    for f in files:
        if len(f) <= len(ckpt_name) + 5:
            continue
        if f[:len(ckpt_name)] == ckpt_name and f[-4:] == '.pkl' and ('best' not in f):
            number = f[len(ckpt_name) + 1:-4]
            iterations.append(int(number))
    num_ckpts = len(iterations)
    if len(iterations) == 0:
        return (-1, num_ckpts) if return_num_ckpts else -1
    if mode == 'max':
        result = max(iterations)
    elif mode == 'all':
        result = sorted(iterations, reverse=True)
    elif mode == 'best':
        eval_file_name = os.path.join(path, '../../eval_result/gathered_eval_result.pkl')
        with open(eval_file_name, 'rb') as handle:
            data = pickle.load(handle)
        cd = np.array(data['avg_cd'])
        idx = np.argmin(cd)
        result = data['iter'][idx]
        print('Found iteration %d with lowest cd loss %.8f' % (result, cd[idx]))
    else:
        raise ValueError('%s mode is not supported' % mode)
    return (result, num_ckpts) if return_num_ckpts else result


def print_size(net):
    """Print the number of parameters of a network."""
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print(f"{net.__class__.__name__} Parameters: {params}")


def std_normal(size, device='cuda'):
    """Generate standard Gaussian variable on the given device."""
    return torch.normal(0, 1, size=size, device=device)


def find_config_file(file_name):
    """Find config JSON file in a directory."""
    if 'config' in file_name and '.json' in file_name:
        if os.path.isfile(file_name):
            return file_name
        else:
            file_path = os.path.split(file_name)[0]
    else:
        if os.path.isdir(file_name):
            file_path = file_name
        else:
            raise FileNotFoundError('%s does not exist' % file_name)
    files = os.listdir(file_path)
    files = [f for f in files if ('config' in f and '.json' in f)]
    print('Found config files: %s' % files)
    config = files[0]
    number = -1
    for f in files:
        all_numbers = re.findall(r'\d+', f)
        all_numbers = [int(n) for n in all_numbers]
        this_number = max(all_numbers) if all_numbers else -1
        if this_number > number:
            config = f
            number = this_number
    print('Chose config:', config)
    return os.path.join(file_path, config)
