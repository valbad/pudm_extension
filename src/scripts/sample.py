"""Sampling/inference script for PUDM point cloud upsampling.

Usage:
    python -m src.scripts.sample -c configs/PU1K.json --strategy ddpm
    python -m src.scripts.sample -c configs/PU1K.json --strategy flow_matching
"""
import argparse
import os
import torch
from shutil import copyfile

from src.data.dataset import get_dataloader
from src.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from src.generative import get_strategy
from src.scripts.eval import evaluate
from src.utils.config import load_config, print_config, get_strategy_config
from src.utils.misc import set_seed


def main(
    config_file,
    pointnet_config,
    dataset_config,
    diffusion_config,
    strategy,
    hyperparams,
    batch_size,
    phase,
    checkpoint_path=None,
    save_dir='',
    gamma=0.5,
    R=4,
    step=30,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving Path: {save_dir}")

    try:
        copyfile(config_file, os.path.join(save_dir, os.path.split(config_file)[1]))
    except Exception:
        pass

    # Move hyperparams to GPU
    for key in hyperparams:
        if key != "T" and isinstance(hyperparams[key], torch.Tensor):
            hyperparams[key] = hyperparams[key].cuda()

    net = PointNet2CloudCondition(pointnet_config).cuda()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"---- Loaded checkpoint: {checkpoint_path} ----")

    dataset_config['batch_size'] = batch_size * torch.cuda.device_count()
    dataset_config['eval_batch_size'] = batch_size * torch.cuda.device_count()
    testloader = get_dataloader(dataset_config, phase=phase)

    CD_loss, HD_loss, P2F_loss, total_meta, metrics = evaluate(
        net=net,
        testloader=testloader,
        strategy=strategy,
        hyperparams=hyperparams,
        print_every_n_steps=diffusion_config["T"] // 5,
        scale=dataset_config['scale'],
        compute_cd=True,
        return_all_metrics=True,
        R=R,
        npoints=dataset_config['npoints'],
        T=diffusion_config["T"],
        step=step,
        save_dir=save_dir,
        gamma=gamma,
        mesh_path=f"{dataset_config['data_dir']}/{phase}/mesh"
    )

    print("{} X :: {}->{} CD: {} HD: {} P2F: {}".format(
        R, dataset_config['npoints'], dataset_config['npoints'] * R,
        CD_loss, HD_loss, P2F_loss
    ), flush=True)

    return CD_loss, HD_loss, P2F_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from trained PUDM model")
    parser.add_argument('-d', '--dataset', type=str, default='PUGAN')
    parser.add_argument('-s', '--strategy', type=str, default='ddpm',
                        choices=['ddpm', 'flow_matching'])
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--R', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=14)
    parser.add_argument('-p', '--phase', type=str, default='test')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.config is None:
        args.config = f"configs/{args.dataset}.json"
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.dataset.lower()}.pkl"
    if args.save_dir is None:
        args.save_dir = f"outputs/{args.dataset.lower()}"

    config = load_config(args.config)
    print_config(config)

    strategy = get_strategy(args.strategy)
    print(f"Using strategy: {strategy.name}")

    train_config = config["train_config"]
    pointnet_config = config["pointnet_config"]
    strategy_config = get_strategy_config(config, args.strategy)

    if train_config['dataset'] == 'PU1K':
        dataset_config = config["pu1k_dataset_config"]
    elif train_config['dataset'] == 'PUGAN':
        dataset_config = config["pugan_dataset_config"]
    else:
        raise ValueError(f"Dataset {train_config['dataset']} not supported")

    hyperparams = strategy.compute_hyperparams(**strategy_config)

    with torch.no_grad():
        main(
            config_file=args.config,
            pointnet_config=pointnet_config,
            dataset_config=dataset_config,
            diffusion_config=strategy_config,
            strategy=strategy,
            hyperparams=hyperparams,
            batch_size=args.batch_size,
            phase=args.phase,
            checkpoint_path=args.checkpoint,
            save_dir=args.save_dir,
            gamma=args.gamma,
            R=args.R,
            step=args.step,
        )
