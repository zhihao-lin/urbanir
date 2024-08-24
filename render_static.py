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
from utils import load_ckpt, save_image, convert_normal
from opt import get_opts
from einops import rearrange
from render import depth2img, semantic2img, render_chunks

def vec2spherical(vec):
    r = np.linalg.norm(vec)
    x, y, z = vec / r
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    spherical = np.array([r, theta, phi])
    return spherical

def spherical2vec(spherical):
    r, theta, phi = spherical
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    vec = np.array([x, y, z])
    return vec

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
        kwargs['load_2d'] = False
    
    if hparams.dataset_name == 'waymo':
        kwargs['frame_start'] = hparams.waymo_start
        kwargs['frame_end']   = hparams.waymo_end
        kwargs['sun_dir'] = hparams.waymo_sun
        kwargs['load_2d'] = False

    if hparams.embed_a:
        dataset_train = dataset(split='train', load_2d=False, **kwargs)
        embedding_a = torch.nn.Embedding(len(dataset_train.poses), hparams.embed_a_len).cuda()
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["model", "msk_model"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())     

    split = 'test' 
    dataset = dataset(split=split, **kwargs)
    poses = dataset.poses
    render_traj_rays = dataset.get_path_rays(poses)

    w, h = dataset.img_wh
    K = dataset.K
    # light source
    relight_idx = hparams.relight_idx 
    relight = (relight_idx != '')
    frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/{relight_idx}'
    os.makedirs(frames_dir, exist_ok=True)
    light_config = hparams.light_config
    light = get_light_model(light_config)
    os.system('cp {} {}'.format(light_config, frames_dir))
    os.system('cp render_static.py {}'.format(frames_dir))
    os.system('cp scripts/relight.sh {}'.format(frames_dir))
    light = light.cuda()
    load_ckpt(light, ckpt_path, model_name='light')
    print('Ambient:', light.ambient)

    ### set up light config ###
    sun_start = np.array([ 0.8826, -0.0316,  0.4691])
    r0, theta0, phi0 = vec2spherical(sun_start)
    theta1, phi1 = 0.95, -0.5

    ###########################

    render_frame_idx = 0
    video_len = hparams.video_len
    frame_series = []
    for img_idx in trange(video_len):
        ### change lighting ###
        a = img_idx / video_len
        theta = (1-a)*theta0 + a*theta1
        phi = (1-a)*phi0 + a*phi1
        sun_dir = spherical2vec([1, theta, phi])
        light.change_direction(sun_dir)
        #######################
        rays = render_traj_rays[render_frame_idx][:, :6].cuda()
        pose = poses[render_frame_idx]
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


    path = os.path.join(frames_dir, 'relight_rgb.mp4')
    imageio.mimsave(path,
                    frame_series,
                    fps=30, macro_block_size=1)

if __name__ == '__main__':
    hparams = get_opts()
    render_for_test(hparams)