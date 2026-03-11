"""Example inference script: upsample a single point cloud file.

Usage:
    python -m src.scripts.example_sample -c configs/PU1K.json --strategy ddpm --example_file input.xyz
"""
import os
import argparse
import numpy as np
import torch
import time

from src.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from src.generative import get_strategy
from src.generative.ddpm import DDPMStrategy
from src.utils.config import load_config, print_config
from src.utils.misc import set_seed
from src.utils.pc_utils import pc_normalize, numpy_to_pc


def evaluate_example(
    net,
    strategy,
    hyperparams,
    example_file,
    print_every_n_steps=200,
    scale=1,
    R=4,
    T=1000,
    step=30,
    save_dir="./test/example",
    gamma=0.5,
    normalization=True,
):
    import open3d

    pc = open3d.io.read_point_cloud(example_file)
    condition = np.asarray(pc.points, dtype=np.float32)

    if normalization:
        condition = pc_normalize(condition)

    condition = condition * scale
    condition = torch.from_numpy(condition).unsqueeze(0).cuda()
    npoints = condition.shape[1]
    label = torch.ones(1, dtype=torch.long).cuda() * (R - 1)

    net.reset_cond_features()
    start_time = time.time()

    if isinstance(strategy, DDPMStrategy) and step < T:
        generated, condition_pre, z = strategy.sample_ddim(
            net=net,
            size=(1, npoints * R, 3),
            hyperparams=hyperparams,
            label=label,
            condition=condition,
            R=R,
            gamma=gamma,
            step=step,
        )
    else:
        sample_kwargs = {}
        if strategy.name == 'FlowMatching':
            sample_kwargs['num_steps'] = step
        generated, condition_pre, z = strategy.sample(
            net=net,
            size=(1, npoints * R, 3),
            hyperparams=hyperparams,
            label=label,
            condition=condition,
            R=R,
            gamma=gamma,
            print_every_n_steps=print_every_n_steps,
            **sample_kwargs,
        )

    elapsed = time.time() - start_time
    print(f"Generation time: {elapsed:.2f}s")

    generated = generated / scale
    generated_np = generated[0].detach().cpu().numpy()
    out_path = os.path.join(save_dir, os.path.basename(example_file))
    open3d.io.write_point_cloud(out_path, numpy_to_pc(generated_np))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsample a single point cloud")
    parser.add_argument('-d', '--dataset', type=str, default='PUGAN')
    parser.add_argument('-s', '--strategy', type=str, default='ddpm',
                        choices=['ddpm', 'flow_matching'])
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--example_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs/example')
    parser.add_argument('--R', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--normalization', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.config is None:
        args.config = f"configs/{args.dataset}.json"
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.dataset.lower()}.pkl"

    config = load_config(args.config)
    strategy = get_strategy(args.strategy)
    hyperparams = strategy.compute_hyperparams(**config["diffusion_config"])

    # Move to GPU
    for key in hyperparams:
        if key != "T" and isinstance(hyperparams[key], torch.Tensor):
            hyperparams[key] = hyperparams[key].cuda()

    net = PointNet2CloudCondition(config["pointnet_config"]).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded: {args.checkpoint}")

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        evaluate_example(
            net=net,
            strategy=strategy,
            hyperparams=hyperparams,
            example_file=args.example_file,
            print_every_n_steps=config["diffusion_config"]["T"] // 5,
            scale=1,
            R=args.R,
            T=config["diffusion_config"]["T"],
            step=args.step,
            save_dir=args.save_dir,
            gamma=args.gamma,
            normalization=args.normalization,
        )
