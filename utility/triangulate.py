import numpy as np
import torch
from matplotlib import pyplot as plt
from datasets.ray_utils import get_rays
from datasets.kitti360 import KittiDataset

def get_sample_index(
    pixel_h, pixel_w,
    image_h, image_w
):
    index = pixel_h * image_w + pixel_w
    return int(index)

def main():
    # check pixel coordinate with: https://pixspy.com/
    image_h = 376
    image_w = 1408
    
    frame_ids = [732, 739]
    
    pixel_coords = [
        [54, 825], [15, 860]
    ]# (h, w)
    kwargs = {
        'seq_id': 0,
        'frame_start': 687,
        'frame_end': 743,
        'test_id': frame_ids
    }
    dataset = KittiDataset(
        '/hdd/datasets/KITTI-360',
        'test',
        ** kwargs
    )

    directions = dataset.directions

    pose = dataset.poses[0]
    rays_o, rays_d = get_rays(directions, pose)
    pixel_h, pixel_w = pixel_coords[0]
    sample_idx = get_sample_index(pixel_h, pixel_w, image_h, image_w)
    o0 = rays_o[sample_idx].numpy()
    ray0 = rays_d[sample_idx].numpy()
    pose = dataset.poses[1]
    rays_o, rays_d = get_rays(directions, pose)
    pixel_h, pixel_w = pixel_coords[1]
    sample_idx = get_sample_index(pixel_h, pixel_w, image_h, image_w)
    o1 = rays_o[sample_idx].numpy()
    ray1 = rays_d[sample_idx].numpy()

    vertical = np.cross(ray0, ray1)
    normal0 = np.cross(ray0, vertical)
    normal1 = np.cross(ray1, vertical)
    s0 = np.sum(normal1 * (o1 - o0)) / np.sum(normal1 * ray0)
    s1 = np.sum(normal0 * (o0 - o1)) / np.sum(normal0 * ray1)
    p0_world = o0 + s0 * ray0
    p1_world = o1 + s1 * ray1
    point_3d = (p0_world + p1_world)/2
    print('Intersected at:', point_3d)
    print('Sun direction:', dataset.sun_dir)
    print('Up  direction:', dataset.up_dir)


    

if __name__ == '__main__':
    main()