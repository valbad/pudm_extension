"""Training script for PUDM point cloud upsampling.

Usage:
    python -m src.scripts.train -c configs/PU1K.json --strategy ddpm
    python -m src.scripts.train -c configs/PU1K.json --strategy flow_matching
"""
import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from shutil import copyfile

from src.data.dataset import get_dataloader
from src.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from src.generative import get_strategy
from src.utils.config import load_config, print_config, get_strategy_config
from src.utils.misc import set_seed


def split_data(data):
    label = data['label'].cuda()
    X = data['complete'].cuda()
    condition = data['partial'].cuda()
    return X, condition, label


def train(
    config_file,
    model_path,
    strategy,
    hyperparams,
    pointnet_config,
    trainset_config,
    dataset,
    root_directory,
    output_directory,
    tensorboard_directory,
    n_epochs,
    epochs_per_ckpt,
    iters_per_logging,
    learning_rate,
):
    local_path = dataset
    tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))
    output_directory = os.path.join(root_directory, local_path, output_directory)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    try:
        copyfile(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
    except Exception:
        print('Config file already in output directory')

    print("output directory:", output_directory, flush=True)

    # Move hyperparams to GPU
    for key in hyperparams:
        if key != "T" and isinstance(hyperparams[key], torch.Tensor):
            hyperparams[key] = hyperparams[key].cuda()

    trainloader = get_dataloader(trainset_config)

    net = PointNet2CloudCondition(pointnet_config).cuda()
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    time0 = time.time()
    epoch = 0

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        time0 -= checkpoint['training_time_seconds']
        print(f"---- Loaded checkpoint: {model_path} ----")
    except Exception:
        print('No valid checkpoint found, training from scratch.', flush=True)

    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)
    loss_function = nn.MSELoss()
    n_iter = epoch * loader_len

    while n_iter < n_iters + 1:
        epoch += 1
        for data in trainloader:
            X, condition, label = split_data(data)
            optimizer.zero_grad()

            loss = strategy.training_loss(
                net, loss_function, X, hyperparams,
                label=label, condition=condition,
            )

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                cprint(
                    "[{}]\tepoch({}/{})\titer({}/{})\t{} loss: {:.6f}".format(
                        time.ctime(), epoch, n_epochs, n_iter, n_iters,
                        strategy.name, loss.item()
                    ),
                    "blue"
                )
                tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)

            n_iter += 1

        if epoch % epochs_per_ckpt == 0:
            checkpoint_name = 'pointnet_ckpt_{}_{:.6f}.pkl'.format(epoch, loss.item())
            checkpoint_path = os.path.join(output_directory, checkpoint_name)
            torch.save({
                'iter': n_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_time_seconds': int(time.time() - time0),
                'epoch': epoch,
                'strategy': strategy.name,
            }, checkpoint_path)
            cprint(f"---- Saved: {checkpoint_path} ----", "red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PUDM point cloud upsampling")
    parser.add_argument('-d', '--dataset', type=str, default='PUGAN')
    parser.add_argument('-s', '--strategy', type=str, default='ddpm',
                        choices=['ddpm', 'flow_matching'],
                        help='Generative strategy to use')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('-m', '--model_path', type=str, default='',
                        help='Path to checkpoint for resuming')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for condition reconstruction loss')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Condition interpolation weight')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.config is None:
        args.config = f"configs/{args.dataset}.json"

    config = load_config(args.config)
    print('Configuration:')
    print_config(config)

    # Get strategy
    strategy = get_strategy(args.strategy)
    print(f"Using generative strategy: {strategy.name}")

    # Parse configs
    train_config = config["train_config"]
    pointnet_config = config["pointnet_config"]
    strategy_config = get_strategy_config(config, args.strategy)

    if train_config['dataset'] == 'PU1K':
        trainset_config = config["pu1k_dataset_config"]
    elif train_config['dataset'] == 'PUGAN':
        trainset_config = config['pugan_dataset_config']
    else:
        raise ValueError(f"Dataset {train_config['dataset']} not supported")

    # Compute strategy-specific hyperparameters
    hyperparams = strategy.compute_hyperparams(**strategy_config)

    if not args.model_path:
        args.model_path = f"exp_{args.dataset.lower()}/{args.dataset}/logs/checkpoint/"

    train(
        config_file=args.config,
        model_path=args.model_path,
        strategy=strategy,
        hyperparams=hyperparams,
        pointnet_config=pointnet_config,
        trainset_config=trainset_config,
        **train_config
    )
