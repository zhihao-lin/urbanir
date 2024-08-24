import numpy as np
import os 
import cv2
import json
from PIL import Image
import math
import pytz
from datetime import datetime
import pvlib
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from justpfm import justpfm as jpfm
from .ray_utils import *
from .base import BaseDataset

class WaymoDataset(BaseDataset):
    def __init__(self, root_dir, split, nvs=False, downsample=1.0, load_2d=True, generate_render_path=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        self.nvs = nvs # exclude testing frames in training
        self.generate_render_path = generate_render_path

        dir_rgb = os.path.join(root_dir, 'image_0')
        dir_sem = os.path.join(root_dir, 'intrinsic_0', 'semantic')
        dir_normal = os.path.join(root_dir, 'intrinsic_0', 'normal')
        dir_shadow = os.path.join(root_dir, 'intrinsic_0', 'shadow')
        dir_depth = os.path.join(root_dir, 'intrinsic_0', 'depth')

        transform_path = os.path.join(root_dir, 'transforms.json')
        with open(transform_path, 'r') as file:
            transform = json.load(file)

        K = np.array([
            [transform['fl_x'], 0, transform['cx']],
            [0, transform['fl_y'], transform['cy'] ],
            [0, 0, 1]
        ])
        K[:2] *= downsample
        self.K = K
        w, h = int(transform['w']*downsample), int(transform['h']*downsample)
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

        # Extrinsics
        frames = transform['frames']
        ids = np.array([frames[i]['colmap_im_id'] for i in range(len(frames))])
        poses = np.array([frames[i]['transform_matrix'] for i in range(len(frames))])
        arg_sort = np.argsort(ids)
        poses = poses[arg_sort][:, :3] # (n, 3, 4) OpenGL, c2w
        poses[:, :, 1:3] *= -1  # (n, 3, 4) OpenCV, c2w

        frame_start = kwargs.get('frame_start', 0)
        frame_end   = kwargs.get('frame_end', 100)
        frame_id = np.arange(frame_start, frame_end)
        self.setup_poses(poses, frame_id)
        sun_dir = kwargs.get('sun_dir', None)
        self.estimate_sunlight(sun_dir)
        
        # self.sunlight_from_2d(cam2world_0, cam2world_1, illum_0, illum_1, frame_id)
        print('#frames = {}'.format(len(frame_id)))
        print('frame_id:', frame_id)
        
        if load_2d:
            print('Load RGB ...')
            rgb = self.read_rgb(dir_rgb, frame_id)
            self.rays = torch.FloatTensor(rgb)
            if self.split == 'train':
                print('Load Semantic ...')
                sem = self.read_semantics(dir_sem, frame_id)
                self.labels = torch.LongTensor(sem)
                print('Load Normal ...')
                normal = self.read_normal(dir_normal, frame_id)
                self.normals = torch.FloatTensor(normal)
                print('Load Shadow ...')
                shadow = self.read_shadow(dir_shadow, frame_id)
                self.shadows = torch.FloatTensor(shadow)
                # print('Load Deshadow ...')
                # deshadow = self.read_rgb(dir_deshadow, frame_id)
                # self.rgb_deshadow = torch.FloatTensor(deshadow)
                print('Load Depth ...')
                depth = self.read_depth(dir_depth, frame_id)
                self.depths = torch.FloatTensor(depth)
    
    def setup_poses(self, cam2world, frame_id):
        cam2world = cam2world[frame_id]

        pos = cam2world[:, :, -1]
        forward = pos[-1] - pos[0]
        forward = forward / np.linalg.norm(forward)
        xyz_min = np.min(pos, axis=0)
        xyz_max = np.max(pos, axis=0)
        center = (xyz_min + xyz_max) / 2
        scale  = np.max(xyz_max - xyz_min) / 2
        self.scale = scale

        pos = (pos - center.reshape(1, -1)) / scale
        pos = pos - forward.reshape(1, -1) * 0.5
        cam2world[:, :, -1] = pos
        self.poses = torch.FloatTensor(cam2world)
        
        if self.generate_render_path:
            render_c2w = generate_interpolated_path(cam2world, 4)[:400]
            self.render_c2w = torch.FloatTensor(render_c2w)
            self.render_traj_rays = self.get_path_rays(render_c2w)

    def get_path_rays(self, render_c2w):
        rays = {}
        print(f'Loading {len(render_c2w)} camera path ...')
        for idx in range(len(render_c2w)):
            c2w = np.array(render_c2w[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

    def read_rgb(self, dir_rgb, frame_id):
        paths = sorted([os.path.join(dir_rgb, f) for f in os.listdir(dir_rgb) if f.endswith('.png')])
        rgb_list = []
        for i in frame_id:
            path = paths[i]
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_wh)
            img = (img / 255.0).astype(np.float32)
            rays = img.reshape(-1, 3)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list

    def read_depth(self, dir_depth, frame_id):
        paths = sorted([os.path.join(dir_depth, f) for f in os.listdir(dir_depth) if f.endswith('.pfm')])
        depth_list = []
        for i in frame_id:
            path = paths[i]
            d_inv = jpfm.read_pfm(file_name=path)
            # process 
            d_inv = np.clip(d_inv/d_inv.max(), 0.05, 1)
            depth = 1 / d_inv 
            depth = (depth - depth.min()) / (depth.max() - depth.min())

            depth = cv2.resize(depth, self.img_wh)
            depth = depth.astype(np.float32).flatten()
            depth_list.append(depth)
        depth_list = np.stack(depth_list)
        return depth_list
    
    def read_semantics(self, dir_sem, frame_id):
        paths = sorted([os.path.join(dir_sem, f) for f in os.listdir(dir_sem) if f.endswith('.pgm')])
        label_list = []
        for i in frame_id:
            path = paths[i]
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, self.img_wh)
            label = label.flatten()
            label_list.append(label)
        label_list = np.stack(label_list)
        return label_list
    
    def read_normal(self, dir_normal, frame_id):
        paths = sorted([os.path.join(dir_normal, f) for f in os.listdir(dir_normal) if f.endswith('.npy')])
        poses = self.poses.numpy()
        normal_list = []
        for c2w, i in zip(poses, frame_id):
            path = paths[i]
            img = np.load(path).transpose(1, 2, 0)
            img = cv2.resize(img, self.img_wh)
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

    def read_shadow(self, dir_shadow, frame_id):
        paths = sorted([os.path.join(dir_shadow, f) for f in os.listdir(dir_shadow) if f.endswith('.png')])
        shadow_list = []
        for i in frame_id:
            path =  paths[i]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_wh)
            img = (img / 255).astype(np.float32).flatten()
            shadow_list.append(img)
        shadow_list = np.stack(shadow_list)
        return shadow_list
    
    def estimate_sunlight(self, sun_dir):
        self.sun_dir = torch.FloatTensor(sun_dir)
        self.sun_dir /= torch.norm(self.sun_dir)
        downs = self.poses.numpy()[:, :, 1]
        self.up_dir  = torch.FloatTensor(-np.mean(downs, axis=0))
        self.up_dir /= torch.norm(self.up_dir)