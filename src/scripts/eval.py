"""Evaluation script for PUDM point cloud upsampling.

Computes Chamfer Distance, Hausdorff Distance, and optionally P2F metrics.
Works with any GenerativeStrategy (DDPM, Flow Matching, etc.).
"""
import os
import numpy as np
import torch
import shutil
import time

from src.generative.ddpm import DDPMStrategy
from src.metrics.chamfer3d import chamfer_3DDist, hausdorff_distance
from src.utils.misc import AverageMeter
from src.utils.pc_utils import numpy_to_pc


def evaluate(
    net,
    testloader,
    strategy,
    hyperparams,
    print_every_n_steps=200,
    scale=1,
    compute_cd=True,
    return_all_metrics=False,
    R=4,
    npoints=2048,
    gamma=0.5,
    T=1000,
    step=30,
    mesh_path="",
    p2f_root="../evaluation_code",
    save_dir="./test/xys",
    save_xyz=True,
    save_sp=True,
    save_z=False,
    save_condition=False,
    save_gt=False,
    save_mesh=False,
    p2f=False,
):
    import open3d

    CD_meter = AverageMeter()
    HD_meter = AverageMeter()
    P2F_meter = AverageMeter()
    total_len = len(testloader)

    total_meta = torch.rand(0).cuda().long()
    metrics = {
        'cd_distance': torch.rand(0).cuda(),
        'h_distance': torch.rand(0).cuda(),
        'cd_p': torch.rand(0).cuda(),
    }

    cd_module = chamfer_3DDist()
    total_time = 0
    cd_result = 0
    times = 0

    print(f"**** {npoints} -----> {npoints * R} ****")

    for idx, data in enumerate(testloader):
        label = data['label'].cuda()
        condition = data['partial'].cuda()
        gt = data['complete'].cuda()

        batch, num_points, _ = gt.shape
        net.reset_cond_features()

        start_time = time.time()

        # Use strategy's sampling method
        # For DDPM, use DDIM if step < T
        if isinstance(strategy, DDPMStrategy) and step < T:
            generated_data, condition_pre, z = strategy.sample_ddim(
                net=net,
                size=(batch, num_points, 3),
                hyperparams=hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma,
                step=step,
            )
        else:
            sample_kwargs = {}
            if hasattr(strategy, 'name') and strategy.name == 'FlowMatching':
                sample_kwargs['num_steps'] = step
            generated_data, condition_pre, z = strategy.sample(
                net=net,
                size=(batch, num_points, 3),
                hyperparams=hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma,
                print_every_n_steps=print_every_n_steps,
                **sample_kwargs,
            )

        generation_time = time.time() - start_time
        times += generation_time
        total_time += generation_time

        generated_data = generated_data / scale
        gt = gt / scale
        torch.cuda.empty_cache()

        if compute_cd:
            cd_p, dist, _, _ = cd_module(generated_data, gt)
            dist = (cd_p + dist) / 2.0
            cd_loss = dist.mean().detach().cpu().item()
        else:
            dist = torch.zeros(generated_data.shape[0], device=generated_data.device)
            cd_p = dist
            cd_loss = 0.0

        cd_result += torch.sum(cd_p).item()

        hd_cost = hausdorff_distance(generated_data, gt)
        hd_loss = hd_cost.mean().detach().cpu().item()

        p2f_loss = 0
        names = data.get('name', [])
        if p2f and names:
            global_p2f = []
            for name in names:
                p2f_path = os.path.join(p2f_root, f"{name}_point2mesh_distance.xyz")
                if os.path.exists(p2f_path):
                    point2mesh_distance = np.loadtxt(p2f_path).astype(np.float32)
                    if point2mesh_distance.size > 0:
                        global_p2f.append(point2mesh_distance[:, 3])
            if global_p2f:
                global_p2f = np.concatenate(global_p2f, axis=0)
                p2f_loss = np.nanmean(global_p2f)

        total_meta = torch.cat([total_meta, label])
        metrics['cd_distance'] = torch.cat([metrics['cd_distance'], dist])
        metrics['h_distance'] = torch.cat([metrics['h_distance'], hd_cost])
        metrics['cd_p'] = torch.cat([metrics['cd_p'], cd_p])

        CD_meter.update(cd_loss, n=batch)
        HD_meter.update(hd_loss, n=batch)
        P2F_meter.update(p2f_loss, n=batch)

        print(
            'progress [%d/%d] %.4f (%d samples) CD %.8f HD %.8f P2F %.8f time %.2fs total %.2fs'
            % (idx, total_len, idx / total_len, batch,
               CD_meter.avg, HD_meter.avg, P2F_meter.avg,
               generation_time, total_time),
            flush=True,
        )

        if save_xyz and names:
            save_path = save_dir
            generated_np = generated_data.detach().cpu().numpy()
            condition_pre_np = condition_pre.detach().cpu().numpy() if condition_pre is not None else None
            z_np = z.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            condition_np = data['partial'].numpy()

            for i in range(len(generated_np)):
                name = names[i]
                # Save generated dense point cloud
                pc = numpy_to_pc(generated_np[i])
                open3d.io.write_point_cloud(os.path.join(save_path, f"{name}.xyz"), pc)

                if save_sp and condition_pre_np is not None:
                    pc = numpy_to_pc(condition_pre_np[i])
                    open3d.io.write_point_cloud(os.path.join(save_path, f"{name}_sp.xyz"), pc)

                if save_mesh and mesh_path:
                    src = os.path.join(mesh_path, f"{name}.off")
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(save_path, f"{name}.off"))

                if save_z:
                    pc = numpy_to_pc(z_np[i])
                    open3d.io.write_point_cloud(os.path.join(save_path, f"{name}_z.xyz"), pc)

                if save_gt:
                    pc = numpy_to_pc(gt_np[i])
                    open3d.io.write_point_cloud(os.path.join(save_path, f"{name}_gt.xyz"), pc)

                if save_condition:
                    pc = numpy_to_pc(condition_np[i])
                    open3d.io.write_point_cloud(os.path.join(save_path, f"{name}_condition.xyz"), pc)

    total_meta = total_meta.detach().cpu().numpy()
    print(f"Total inference time: {times:.2f}s")

    if return_all_metrics:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics
    else:
        return CD_meter.avg, HD_meter.avg, P2F_meter.avg, total_meta, metrics['cd_distance']
