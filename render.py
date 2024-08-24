import os
import torch
import imageio
import numpy as np
import cv2
import math 
from PIL import Image
from tqdm import trange
from models.networks import NGP
from models.rendering import render
from models.shading import SunLight, get_light_model
from models.const import SKY_LABEL
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt, save_image, convert_normal, FrameEmbedding
from opt import get_opts
from einops import rearrange

def depth2img(depth, scale=16):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_MAGMA)
    # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB) # for TURBO
    return depth_img

def semantic2img(sem_label, classes):
    level = 1/(classes-1)
    sem_color = level * sem_label
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color

def render_chunks(model, rays_o, rays_d, chunk_size, **kwargs):
    chunk_n = math.ceil(rays_o.shape[0]/chunk_size)
    results = {}
    for i in range(chunk_n):
        rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
        rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
        ret = render(model, rays_o_chunk, rays_d_chunk, **kwargs)
        for k in ret:
            if k not in results:
                results[k] = []
            results[k].append(ret[k])
    for k in results:
        if k in ['total_samples']:
            continue
        results[k] = torch.cat(results[k], 0)
    return results

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len, classes=hparams.num_classes).cuda()
    if hparams.ckpt_load:
        ckpt_path = hparams.ckpt_load
    else: 
        ckpt_path = os.path.join('ckpts', hparams.dataset_name, hparams.exp_name, 'last_slim.ckpt')

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model', 'density_grid', 'grid_coords'])
    print('Loaded checkpoint: {}'.format(ckpt_path))   
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_traj': hparams.render_traj,
            'anti_aliasing_factor': hparams.anti_aliasing_factor}
    
    if hparams.dataset_name == 'kitti':
        kwargs['seq_id'] = hparams.kitti_seq
        kwargs['frame_start'] = hparams.kitti_start
        kwargs['frame_end'] = hparams.kitti_end
        kwargs['test_id'] = hparams.kitti_test_id
        kwargs['nvs'] = hparams.nvs
        kwargs['load_2d'] = False
    
    if hparams.dataset_name == 'waymo':
        kwargs['frame_start'] = hparams.waymo_start
        kwargs['frame_end']   = hparams.waymo_end
        kwargs['sun_dir'] = hparams.waymo_sun
        kwargs['load_2d'] = False
    
    if hparams.embed_a:
        dataset_train = dataset(split='train', **kwargs)
        frame_embed = FrameEmbedding(hparams.embed_a_len, dataset_train.poses, ckpt_path)

    poses, render_traj_rays = None, None
    if hparams.render_traj:
        embed_mode = 'first'
        dataset = dataset(split='test', generate_render_path=True, **kwargs)
        render_traj_rays = dataset.render_traj_rays
        poses = dataset.render_c2w
    else:
        embed_mode = 'mean'
        split = 'test' if hparams.nvs else 'train'
        dataset = dataset(split=split, **kwargs)
        poses = dataset.poses
        render_traj_rays = dataset.get_path_rays(poses)
    print('Embedding mode:', embed_mode)

    w, h = dataset.img_wh
    K = dataset.K
    # light source
    relight_idx = hparams.relight_idx 
    relight = (relight_idx != '')
    if not relight:
        dir_name = 'frames' if hparams.render_traj else 'nvs' if hparams.nvs else 'train'
        frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/{dir_name}'
        os.makedirs(frames_dir, exist_ok=True)
        sun_config = {
            'direction': dataset.sun_dir,
            'up': dataset.up_dir,
            'sky_height': hparams.sky_height,
            'near_dist_T': hparams.near_dist_T,
        }
        light = SunLight(sun_config)

    else:
        frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/{relight_idx}'
        os.makedirs(frames_dir, exist_ok=True)
        light_config = hparams.light_config
        light = get_light_model(light_config)
        command = 'cp {} {}'.format(light_config, frames_dir)
        os.system(command)
    light = light.cuda()
    load_ckpt(light, ckpt_path, model_name='light')
    print('Ambient:', light.ambient)
    
    frame_series = []
    albedo_series = []
    depth_raw_series = []
    depth_series = []
    normal_series = []
    normal_raw_series = []
    semantic_series = []
    visibility_series = []
    visibility_T_series = []
    shading_series = []

    if hasattr(dataset, 'sun_dir'):
        sun_dir = dataset.sun_dir.numpy()

    for img_idx in trange(len(render_traj_rays)):
        rays = render_traj_rays[img_idx][:, :6].cuda()
        pose = poses[img_idx]
        render_kwargs = {
            'img_idx': img_idx,
            'pose': pose,
            'K': K,
            'test_time': True,
            'T_threshold': 1e-2,
            'render_rgb': hparams.render_rgb,
            'render_depth': hparams.render_depth,
            'render_normal': hparams.render_normal,
            'render_sem': hparams.render_semantic,
            'num_classes': hparams.num_classes,
            'relight': relight,
            'img_wh': dataset.img_wh,
            'anti_aliasing_factor': hparams.anti_aliasing_factor,
            'light': light
        }
        if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            embedding_a = frame_embed(pose, mode=embed_mode).cuda()
            render_kwargs['embedding_a'] = embedding_a

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        results = {}
        chunk_size = hparams.chunk_size
        if chunk_size > 0:
            results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)
        else:
            results = render(model, rays_o, rays_d, **render_kwargs)

        semantic = rearrange(results['semantic'].cpu().numpy(), '(h w) -> h w', h=h)
        sky_mask = (semantic == SKY_LABEL)

        if hparams.render_rgb:
            rgb_frame = None
            if hparams.anti_aliasing_factor > 1.0:
                h_new = int(h*hparams.anti_aliasing_factor)
                rgb_frame = rearrange(results['rgb_shading'].cpu().numpy(), '(h w) c -> h w c', h=h_new)
                rgb_frame = np.clip(rgb_frame, 0, 1)
                rgb_frame = Image.fromarray((rgb_frame*255).astype(np.uint8)).convert('RGB')
                rgb_frame = np.array(rgb_frame.resize((w, h), Image.Resampling.BICUBIC))
            else:
                rgb_frame = rearrange(results['rgb_shading'].cpu().numpy(), '(h w) c -> h w c', h=h)
                rgb_frame = np.clip(rgb_frame, 0, 1)
                rgb_frame = (rgb_frame*255).astype(np.uint8)
            if relight and hparams.add_flares:
                rgb_frame = light.add_flares(rgb_frame, **render_kwargs)
            frame_series.append(rgb_frame)
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-rgb.png'.format(img_idx)), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        if hparams.render_albedo and not relight:
            albedo = rearrange(results['albedo'].cpu().numpy(), '(h w) c -> h w c', h=h)
            albedo[sky_mask] = 1.0
            save_image(albedo, os.path.join(frames_dir, '{:0>3d}-albedo.png'.format(img_idx)))
            albedo_series.append((albedo*255).astype(np.uint8))

        if hparams.render_depth and not relight:
            depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
            depth_raw_series.append(depth_raw)
            depth = depth2img(depth_raw, scale=0.4*hparams.scale)
            depth[sky_mask] = 255
            depth_series.append(cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-depth.png'.format(img_idx)), depth)

        if hparams.render_normal and not relight:
            pose = pose.cpu().numpy()
            normal_pred = rearrange(results['normal_pred'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6
            normal_vis = (convert_normal(normal_pred, pose) + 1)/2
            normal_vis[sky_mask] = 1.0
            save_image(normal_vis, os.path.join(frames_dir, '{:0>3d}-normal.png'.format(img_idx)))
            normal_series.append((255*normal_vis).astype(np.uint8))
            normal_raw = rearrange(results['normal_raw'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6
            normal_vis = (convert_normal(normal_raw, pose) + 1)/2
            normal_vis[sky_mask] = 1.0
            save_image(normal_vis, os.path.join(frames_dir, '{:0>3d}-normal-raw.png'.format(img_idx)))
            normal_raw_series.append((255*normal_vis).astype(np.uint8))

        if hparams.render_semantic and not relight:
            sem_frame = semantic2img(semantic, hparams.num_classes)
            semantic_series.append(sem_frame)

        if hparams.render_visibility and not relight:
            visibility = rearrange(results['visibility'].cpu().numpy(), '(h w) -> h w', h=h)
            visibility[sky_mask] = 1.0
            save_image(visibility, os.path.join(frames_dir, '{:0>3d}-vis.png'.format(img_idx)))
            visibility_series.append((visibility*255).astype(np.uint8))
            visibility_T = rearrange(results['visibility_T'].cpu().numpy(), '(h w) -> h w', h=h)
            visibility_T[sky_mask] = 1.0
            save_image(visibility_T, os.path.join(frames_dir, '{:0>3d}-vis-T.png'.format(img_idx)))
            visibility_T_series.append((visibility_T*255).astype(np.uint8))

        if hparams.render_shading and not relight:
            shading = np.sum(normal_pred * sun_dir[np.newaxis, np.newaxis], axis=-1)
            shading = np.clip(shading, 0, 1)
            shading *= visibility
            shading[sky_mask] = 1.0
            save_image(shading, os.path.join(frames_dir, '{:0>3d}-shading.png'.format(img_idx)))
            shading_series.append((shading*255).astype(np.uint8))
        torch.cuda.synchronize()

    if not hparams.render_traj:
        quit()
    if hparams.render_rgb:
        if not relight:
            path = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_rgb.mp4')
        else:
            path = os.path.join(frames_dir, 'relight_rgb.mp4')
        imageio.mimsave(path,
                        frame_series,
                        fps=30, macro_block_size=1)
    
    if hparams.render_albedo and not relight:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_albedo.mp4'),
                        albedo_series,
                        fps=30, macro_block_size=1)

    if hparams.render_depth and not relight:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_depth.mp4'),
                        depth_series,
                        fps=30, macro_block_size=1)

    if hparams.render_normal and not relight:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_normal.mp4'),
                        normal_series,
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_normal_raw.mp4'),
                        normal_raw_series,
                        fps=30, macro_block_size=1)

    if hparams.render_semantic and not relight:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_semantic.mp4'),
                        semantic_series,
                        fps=30, macro_block_size=1)
        
    if hparams.render_visibility and not relight:
        path = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_visibility.mp4')
        imageio.mimsave(path, visibility_series, fps=30, macro_block_size=1)
        path = os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_visibility_T.mp4')
        imageio.mimsave(path, visibility_T_series, fps=30, macro_block_size=1)
        
    if hparams.render_shading and not relight:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_shading.mp4'),
                        shading_series,
                        fps=30, macro_block_size=1)

if __name__ == '__main__':
    hparams = get_opts()
    render_for_test(hparams)