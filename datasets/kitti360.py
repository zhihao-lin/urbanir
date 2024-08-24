import numpy as np
import os 
import cv2
from PIL import Image
import math
import pytz
from datetime import datetime
import pvlib
import torch
from scipy.spatial.transform import Rotation as R
from justpfm import justpfm as jpfm
from .ray_utils import *
from .base import BaseDataset

class KittiDataset(BaseDataset):
    def __init__(self, root_dir, split, nvs=False, downsample=1.0, load_2d=True, generate_render_path=False, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        self.nvs = nvs # exclude testing frames in training
        self.generate_render_path = generate_render_path

        seq_id = kwargs.get('seq_id', 0)
        dir_seq = '2013_05_28_drive_{:0>4d}_sync'.format(seq_id)
        dir_rgb_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'data_rect')
        dir_rgb_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'data_rect')
        dir_sem_0 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_00/semantic')
        dir_sem_1 = os.path.join(root_dir, 'data_2d_semantics/train', dir_seq, 'image_01/semantic')
        dir_normal_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'normal')
        dir_normal_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'normal')
        dir_shadow_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'shadow')
        dir_shadow_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'shadow')
        dir_depth_0 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_00', 'depth')
        dir_depth_1 = os.path.join(root_dir, 'data_2d_raw', dir_seq, 'image_01', 'depth')
        dir_calib = os.path.join(root_dir, 'calibration')
        dir_poses = os.path.join(root_dir, 'data_poses', dir_seq)
        dir_oxts = os.path.join(root_dir, 'data_poses', dir_seq, 'oxts')

        # Intrinsics
        intrinsic_path = os.path.join(dir_calib, 'perspective.txt')
        K_00 = parse_calib_file(intrinsic_path, 'P_rect_00').reshape(3, 4)
        K_00[:2] *= downsample
        self.K = K_00[:, :-1]
        img_size = parse_calib_file(intrinsic_path, 'S_rect_00')
        w, h = int(img_size[0]), int(img_size[1])
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

        # Extrinsics
        frame_start = kwargs.get('frame_start', 0)
        frame_end   = kwargs.get('frame_end', 100)
        pose_cam_0 = np.genfromtxt(os.path.join(dir_poses, 'cam0_to_world.txt')) #(n, 17)
        frame_id = pose_cam_0[:, 0]
        sample = np.logical_and(frame_id >= frame_start, frame_id <= frame_end)
        frame_id = frame_id[sample].astype(np.int32)

        cam2world_0 = pose_cam_0[sample, 1:].reshape(-1, 4, 4)[:, :3]
        sys2world = np.genfromtxt(os.path.join(dir_poses, 'poses.txt'))
        sys2world = sys2world[sample, 1:].reshape(-1, 3, 4)
        cam2sys_1 = parse_calib_file(os.path.join(dir_calib, 'calib_cam_to_pose.txt'), 'image_01')
        cam2sys_1 = np.concatenate([cam2sys_1.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
        R_rect_01 = parse_calib_file(intrinsic_path, 'R_rect_01').reshape(3, 3)
        R_rect = np.eye(4)
        R_rect[:3:, :3] = np.linalg.inv(R_rect_01)
        cam2world_1 = sys2world @ cam2sys_1 @ R_rect
        test_id = np.array(kwargs['test_id']).astype(np.int32)
        test_id_normalized = []
        for i in range(len(frame_id)):
            if (test_id == frame_id[i]).any():
                test_id_normalized.append(i)
        test_id_normalized = np.array(test_id_normalized)
        self.setup_poses(cam2world_0, cam2world_1, test_id_normalized)
        self.estimate_sunlight(dir_oxts, dir_calib, frame_id, cam2world_0)

        if self.split != 'train':
            frame_id = test_id
        elif self.split == 'train' and self.nvs:
            sample = torch.ones(len(frame_id)).bool()
            sample[test_id_normalized] = False
            frame_id = frame_id[sample]
        print('#frames = 2 * {}'.format(len(frame_id)))
        print('frame_id:', frame_id)
        self.frame_id = frame_id
        
        if load_2d:
            print('Load RGB ...')
            rgb_0 = self.read_rgb(dir_rgb_0, frame_id)
            rgb_1 = self.read_rgb(dir_rgb_1, frame_id)
            self.rays = torch.FloatTensor(np.concatenate([rgb_0, rgb_1], axis=0))
            if self.split == 'train':
                print('Load Semantic ...')
                sem_0 = self.read_semantics(dir_sem_0, frame_id)
                sem_1 = self.read_semantics(dir_sem_1, frame_id)
                self.labels = torch.LongTensor(np.concatenate([sem_0, sem_1], axis=0))
                print('Load Normal ...')
                normal_0 = self.read_normal(dir_normal_0, frame_id)
                normal_1 = self.read_normal(dir_normal_1, frame_id)
                self.normals = torch.FloatTensor(np.concatenate([normal_0, normal_1], axis=0))
                print('Load Shadow ...')
                shadow_0 = self.read_shadow(dir_shadow_0, frame_id)
                shadow_1 = self.read_shadow(dir_shadow_1, frame_id)
                self.shadows = torch.FloatTensor(np.concatenate([shadow_0, shadow_1], axis=0))
                print('Load Depth ...')
                depth_0 = self.read_depth(dir_depth_0, frame_id)
                depth_1 = self.read_depth(dir_depth_1, frame_id)
                self.depths = torch.FloatTensor(np.concatenate([depth_0, depth_1], axis=0))
    
    def setup_poses(self, cam2world_0, cam2world_1, test_id_normalized):
        pos_0 = cam2world_0[:, :, -1]
        pos_1 = cam2world_1[:, :, -1]
        pos = np.concatenate([pos_0, pos_1], axis=0)
        forward = pos_0[-1] - pos_0[0]
        forward = forward / np.linalg.norm(forward)
        xyz_min = np.min(pos, axis=0)
        xyz_max = np.max(pos, axis=0)
        center = (xyz_min + xyz_max) / 2
        scale  = np.max(xyz_max - xyz_min) / 2
        self.scale = scale

        pos = (pos - center.reshape(1, -1)) / scale
        pos = pos - forward.reshape(1, -1) * 0.5
        cam2world = np.concatenate([cam2world_0, cam2world_1], axis=0)
        cam2world[:, :, -1] = pos
        if self.split != 'train':
            n_step = cam2world_0.shape[0]
            test_id = np.concatenate([test_id_normalized, test_id_normalized + n_step])
            cam2world = cam2world[test_id]
        elif self.split == 'train' and self.nvs:
            n_step = cam2world_0.shape[0]
            test_id = np.concatenate([test_id_normalized, test_id_normalized + n_step])
            sample = torch.ones(n_step*2).bool()
            sample[test_id] = False
            cam2world = cam2world[sample]

        self.poses = torch.FloatTensor(cam2world)
        if self.generate_render_path:
            render_c2w = generate_interpolated_path(cam2world, 120)[:400]
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
        rgb_list = []
        for i in frame_id:
            path = os.path.join(dir_rgb, '{:0>10d}.png'.format(i))
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = (img / 255.0).astype(np.float32)
            rays = img.reshape(-1, 3)
            rgb_list.append(rays)
        rgb_list = np.stack(rgb_list)
        return rgb_list
    
    def read_semantics(self, dir_sem, frame_id):
        label_list = []
        for i in frame_id:
            path = os.path.join(dir_sem, '{:0>10d}.png'.format(i))
            label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = self.label_mapping(label.flatten())
            label_list.append(label)
        label_list = np.stack(label_list)
        return label_list

    def label_mapping(self, label):
        # kitti360 label: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
        def get_mask(label, mask_ids):
            mask = np.zeros_like(label) > 1
            for idx in mask_ids:
                mask = np.logical_or(mask, label==idx)
            return mask
        label_new = np.ones_like(label).astype(np.int32)
        label_new[:] = 9    # void
        mask = get_mask(label, [6, 7])
        label_new[mask] = 0 # ground/road
        mask = get_mask(label, [8, 9, 10])
        label_new[mask] = 1 # sidewalk/parking/rail track
        mask = get_mask(label, [11, 12, 13, 14, 15, 16, 34, 35, 36, 42])
        label_new[mask] = 2 # construction
        mask = get_mask(label, [39, 40, 41, 44])
        label_new[mask] = 3 # object
        mask = get_mask(label, [21, 22])
        label_new[mask] = 4 # nature
        mask = get_mask(label, [23])
        label_new[mask] = 5 # sky
        mask = get_mask(label, [24, 25])
        label_new[mask] = 6 # human
        mask = get_mask(label, [19, 20, 26, 27, 28, 29, 30, 31, 32, 33, 43, -1])
        label_new[mask] = 7 # vehicle
        # mask = get_mask(label, [38])
        # label_new[mask] = 10 # light source
        mask = get_mask(label, [17, 18, 37])
        label_new[mask] = 8 # pole
        return label_new
    
    def read_normal(self, dir_normal, frame_id):
        poses = self.poses.numpy()
        normal_list = []
        for c2w, i in zip(poses, frame_id):
            path = os.path.join(dir_normal, '{:0>10d}.npy'.format(i))
            img = np.load(path).transpose(1, 2, 0)
            normal = ((img - 0.5) * 2).reshape(-1, 3)
            normal = normal @ c2w[:,:3].T
            normal_list.append(normal)
        normal_list = np.stack(normal_list)
        return normal_list

    def read_shadow(self, dir_shadow, frame_id):
        shadow_list = []
        for i in frame_id:
            path = os.path.join(dir_shadow, '{:0>10d}.png'.format(i))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = (img / 255).astype(np.float32).flatten()
            shadow_list.append(img)
        shadow_list = np.stack(shadow_list)
        return shadow_list
    
    def read_depth(self, dir_depth, frame_id):
        depth_list = []
        for i in frame_id:
            path = os.path.join(dir_depth, '{:0>10d}.pfm'.format(i))
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

    def estimate_sunlight(self, dir_oxts, dir_calib, frame_id, cam2world):
        with open(os.path.join(dir_oxts, 'timestamps.txt'), 'r') as file:
            time_stamps = file.readlines()
            time_stamps = np.array([line.strip() for line in time_stamps])
            time_stamps = time_stamps[frame_id]
        dir_data = os.path.join(dir_oxts, 'data')
        data_files = sorted([os.path.join(dir_data, path) for path in os.listdir(dir_data)])
        data_files = np.array(data_files)[frame_id]
        cam2sys_0 = parse_calib_file(os.path.join(dir_calib, 'calib_cam_to_pose.txt'), 'image_00')
        R_cam2sys_0 = cam2sys_0.reshape(3, 4)[:3, :3]
        R_sys2cam_0 = R_cam2sys_0.T

        # Solar position estimation: 
        # https://assessingsolar.org/notebooks/solar_position.html
        # world coord: x(east), y(north), z(up)
        # car faces north and level on ground if (roll, pitch, yaw) = (0, 0, 0)
        # kitti360 oxts2pose: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/convertOxtsToPose.py
        # kitti360 device coordinates: https://www.cvlibs.net/datasets/kitti-360/documentation.php

        time_zone = 'Europe/Berlin'
        local_time_zone = pytz.timezone(time_zone) 
        sun_dirs = []
        up_dirs = []
        for i in range(len(frame_id)):
            time = time_stamps[i].split('.')[0]
            time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            localized_time = local_time_zone.localize(time)
            utc_time = localized_time.astimezone(pytz.utc)

            data = np.genfromtxt(data_files[i])
            lat, lon, alt = data[0], data[1], data[2]
            site = pvlib.location.Location(lat, lon, time_zone, alt)
            sun_pose = site.get_solarposition(utc_time)
            theta, phi = math.radians(sun_pose["zenith"].item()), math.radians(sun_pose["azimuth"].item())
            sz = np.cos(theta)
            sy = np.sin(theta) * np.cos(phi)
            sx = np.sin(theta) * np.sin(phi)
            sun = np.array([sx, sy, sz])
            roll  = data[3] # 0 = level, positive = left side up,      range: -pi   .. +pi
            pitch = data[4] # 0 = level, positive = front down,        range: -pi/2 .. +pi/2
            yaw   = data[5] # 0 = east,  positive = counter clockwise, range: -pi   .. +pi
            rx = -pitch # positive = front up
            ry = roll 
            rz = yaw - np.pi/2 # 0 = north
            car2world = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
            world2car = car2world.T
            sun2car = world2car @ sun # x=right, y=forward, z=up
            sun2gps = np.array([sun2car[1], sun2car[0], -sun2car[2]]) # GPS/IMU, x=forward, y=right, z=down
            sun2cam = R_sys2cam_0 @ sun2gps # Opencv: x=right, y=down, z=forward
            # print('Sun:', sun2cam)
            R_cam2world = cam2world[i][:3, :3]
            sun_world = R_cam2world @ sun2cam
            sun_dirs.append(sun_world)

            up = np.array([0, 0, 1])
            up2car = world2car @ up
            up2gps = np.array([up2car[1], up2car[0], -up2car[2]])
            up2cam = R_sys2cam_0 @ up2gps
            # print('Up:', up2cam)
            up_world = R_cam2world @ up2cam
            up_dirs.append(up_world)
            
        sun_dirs = np.stack(sun_dirs)
        sun_dir = torch.FloatTensor(np.mean(sun_dirs, axis=0))
        print('sun dir from gps:', sun_dir / torch.norm(sun_dir))
        self.sun_dir = sun_dir

        up_dirs = np.stack(up_dirs)
        up_dir = torch.FloatTensor(np.mean(up_dirs, axis=0))
        self.up_dir = up_dir
        print('up dir from gps:', up_dir)
     
def parse_calib_file(path, key):
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        if key in line:
            tokens = line.strip().split(' ')
            nums = tokens[1:]
            array = np.array([float(i) for i in nums])
            return array
    return None