from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch 
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
from einops import rearrange
from models.volume_render import volume_render
from models.custom_functions import RayAABBIntersector
from models.const import MAX_SAMPLES, SKY_LABEL, VEHICLE_LABEL
from models.custom_functions import RayMarcher, VolumeRendererVisibility
from models.visibility import sample_visibility_test

def get_light_model(config_path):
    config = OmegaConf.load(config_path)
    mode = config['mode']
    device = config['device']

    if mode == 'sun':
        sun_config = config['sun']
        light_model = SunLight(sun_config, device=device)
    elif mode == 'night':
        night_config = config['night']
        light_model = NightLight(night_config, device)
    else:
        print('[ERROR] No such lighting mode: {}'.format(mode))
        quit()
    return light_model

def normalized_tensor(x, device='cuda'):
    x = torch.FloatTensor(x).to(device)
    x = x / x.norm()
    return x

def surface_points(rays_o, rays_d, depth):
    surface = rays_o + rays_d * depth.unsqueeze(-1)
    return surface

class SunSkyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.ambient = nn.Parameter(torch.FloatTensor([0.30112045, 0.32212885, 0.35714286])) #RGB

        self.sky_dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Frequency",
	                "n_frequencies": 32 
                },
            )
        
        self.sky_rgb_net = \
            tcnn.Network(
                n_input_dims=self.sky_dir_encoder.n_output_dims,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": 'Sigmoid',
                    "n_neurons": 128,
                    "n_hidden_layers": 1,
                }
            )
        
    def forward(self, d):
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.sky_dir_encoder((d+1)/2)
        rgbs = self.sky_rgb_net(d)
        return rgbs

class NightLight(SunSkyModel):
    def __init__(
        self,
        config,
        device
    ):  
        super().__init__()
        self.ambeint_sky = torch.tensor(config['ambient']['sky'], device=device)   
        self.ambient_nonsky = torch.tensor(config['ambient']['nonsky'], device=device)
        self.lights = []
        spotlights_config = config['spotlights']
        for key in list(spotlights_config.keys()):
            light_config = spotlights_config[key]
            light_config['device'] = device
            spotlight = SpotLight(light_config)
            self.lights.append(spotlight)

    def shading_model(
        self,
        model,
        points,
        albedo,
        normal,
        semantic,
        view_d,
        **kwargs
    ):
        shading_all, visible_all = [], []
        for light in self.lights:
            shading, visble = light.get_shading_visibility(
                model, points, normal, 
                semantic, view_d, **kwargs
            )
            shading_all.append(shading)
            visible_all.append(visble)
        shading_all = torch.stack(shading_all)
        visible_all = torch.stack(visible_all)
        shading = torch.sum(shading_all, dim=0)
        visible = torch.sum(visible_all, dim=0)
        rgb = albedo * (shading + self.ambient_nonsky.unsqueeze(0))

        label = torch.max(semantic, dim=-1)[1]
        sky_mask = (label == SKY_LABEL)
        rgb_sky = self(view_d)
        rgb[sky_mask] = rgb_sky[sky_mask] * self.ambeint_sky.unsqueeze(0)
        return rgb, visible

    def add_flares(self, rgb, **kwargs):
        rgb = rgb / 255.0
        for light in self.lights:
            rgb = light.add_flares(rgb, **kwargs)
        rgb = (rgb * 255).astype(np.uint8)
        return rgb
class SpotLight():
    def __init__(
        self,
        config
    ):  
        device = config['device']
        self.position = torch.FloatTensor(config['position']).to(device)
        self.direction = normalized_tensor(config['direction'], device)
        self.color = torch.FloatTensor(config['color']).to(device)
        self.intensity = config['intensity']
        self.spot_exp = config['spot_exp']
        self.near_dist_T = config['near_dist_T']
        # bound the max value in decay function
        self.decay_bound = config['decay_bound'] # 10: light intensity decays with distance > 0.3
        self.emission_r = config['emission_r'] # 0.05
        self.shininess = config['shininess'] # 30
        self.specular_strength = config['specular_strength']
        self.move_with_cam = config['move_with_cam']
        self.device = device
        if 'flare_path' in config:
            img = Image.open(config['flare_path']).convert('RGBA')
            self.flare_rgba = img #(h, w, 4)
        else:
            self.flare_rgba = None
        self.flare_size = config['flare_size'] if 'flare_size' in config else 200
        
    
    def place_at_camera(
        self,
        pose_c2w
    ):
        pose_c2w = pose_c2w.to(self.device)
        cam_t = pose_c2w[:,-1]
        cam_R = pose_c2w[:,:3]
        position = cam_t + cam_R @ self.position
        direction = cam_R @ self.direction
        return position, direction

    def get_tracing_rays(
        self,
        position,
        model,
        points
    ):
        light_d = position[None] - points
        light_dist = torch.norm(light_d, dim=-1) + 1e-6 #(h*w, )
        light_d = light_d / light_dist.unsqueeze(-1)

        NEAR_DISTANCE_T = self.near_dist_T
        _, hits_t, _ = RayAABBIntersector.apply(points, light_d, model.center, model.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE_T), 0, 0] = NEAR_DISTANCE_T
        hits_t = hits_t[:,0,:] 
        return points, light_d, light_dist, hits_t

    def get_shading_visibility(
        self,
        model, 
        points,
        normal,
        semantic,
        view_d,
        **kwargs
    ):
        if self.move_with_cam:
            pose_c2w = kwargs.get('pose', None)
            position, direction = self.place_at_camera(pose_c2w)
        else:
            position, direction = self.position, self.direction
        rays_o, rays_d, light_dist, hits_t = self.get_tracing_rays(position, model, points)

        visibility = sample_visibility_test(model, rays_o, rays_d, hits_t, light_dist, **kwargs)
        decay = torch.clip(1.0/ (light_dist * light_dist), 0, self.decay_bound)  
        spot_focus = torch.sum(rays_d * -direction[None], dim=-1) 
        spot_intensity = torch.clip(spot_focus, 0.0, 1.0) ** self.spot_exp
        light_intensity = self.intensity * visibility * decay * spot_intensity

        label = torch.max(semantic, dim=-1)[1]
        vehicle_mask = (label == VEHICLE_LABEL)

        diffuse = torch.sum(rays_d * normal, dim=-1)
        diffuse = torch.clip(diffuse, 0.0, 1.0)
        diffuse[vehicle_mask] *= 0.1
        
        half_d = F.normalize(-view_d + rays_d, dim=-1)
        normal = F.normalize(normal, dim=-1)
        specular = torch.sum(half_d * normal, dim=-1)
        specular = vehicle_mask * self.specular_strength * (torch.clip(specular, 0.0, 1.0) ** self.shininess)
        emission = (light_dist < self.emission_r)

        shading = (diffuse + specular) * light_intensity + emission
        shading = shading[:, None] * self.color[None]
        temp = specular * light_intensity
        return shading, temp#, visibility

    def add_flares(self, rgb, **kwargs):
        # rgb: numpy array (h, w, 3)
        if self.flare_rgba is None:
            return rgb
        # calculate light position
        Rt_c2w = kwargs.get('pose', None).numpy()
        K = np.array(kwargs.get('K', None))
        R_w2c = Rt_c2w[:3, :3].T
        t_w2c = R_w2c @ -Rt_c2w[:, 3]
        light_w = self.position.cpu().numpy()
        light_c = R_w2c @ light_w + t_w2c
        depth = light_c[-1]
        light_i = light_c @ K.T
        light_i = light_i[:2] / light_i[2] # (x, y)
        # check if it's seen
        h, w, _ = rgb.shape
        x, y = light_i
        vis_x = np.logical_and(x >= 0, x <= w)
        vis_y = np.logical_and(y >= 0, y <= h)
        vis_xy = np.logical_and(vis_x, vis_y)
        vis = np.logical_and(vis_xy, depth > 0)
        if not vis:
            return rgb
        eps = 0.5
        flare_size = eps/(eps + depth) * self.flare_size
        flare_size = int(flare_size)
        flare_rgba = self.flare_rgba.resize((flare_size, flare_size))
        flare_rgba = np.array(flare_rgba) / 255.0 #(flare_size, flare_size, 4)

        left, top = int(x - flare_size/2), int(y - flare_size/2)
        right, bottom = left + flare_size, top + flare_size
        x_start, y_start = max(0, left), max(0, top)
        x_end, y_end = min(w, right), min(h, bottom)
        ov_x_start, ov_y_start = -min(0, left), -min(0, top)
        ov_x_end, ov_y_end = flare_size + min(0, w-right), flare_size + min(0, h-bottom)

        ov_rgba_patch = flare_rgba[ov_y_start:ov_y_end, ov_x_start:ov_x_end]
        ov_rgb_patch = ov_rgba_patch[:, :, :3]
        ov_a_patch = ov_rgba_patch[:, :, 3:4]

        bg_rgb_patch = rgb[y_start:y_end, x_start:x_end]
        bg_a_patch = np.ones_like(ov_a_patch)
        out_a_patch = ov_a_patch + (1 - ov_a_patch) * bg_a_patch
        out_rgb_patch = ov_rgb_patch * ov_a_patch + bg_rgb_patch * (1-ov_a_patch) * bg_a_patch
        out_rgb_patch /= out_a_patch
        rgb[y_start:y_end, x_start:x_end] = out_rgb_patch
        return rgb 
    
class SunLight(SunSkyModel):
    def __init__(
        self, 
        config,
        device='cuda'
    ):
        super().__init__()
        self.direction = normalized_tensor(config['direction'], device)
        self.up = normalized_tensor(config['up'], device)
        self.sky_height = config['sky_height']
        self.near_dist_T = config['near_dist_T']
        sky_shading = config['sky_shading'] if 'sky_shading' in config else [1.0, 1.0, 1.0]
        self.sky_shading = torch.tensor(sky_shading).to(device)
        self.shininess = config['shininess'] if 'shininess' in config else 10
        self.specular_strength = config['specular_strength'] if 'specular_strength' in config else 1.0
        self.device = device
    
    def change_direction(self, direction):
        self.direction = normalized_tensor(direction, self.device)

    def shading_model(
        self,
        model,
        points,
        albedo,
        normal,
        semantic,
        view_d,
        visible=None,
        **kwargs
    ):
        diffuse = torch.sum(normal * self.direction.unsqueeze(0), dim=-1)
        diffuse = torch.clip(diffuse, 0.0, 1.0)
        specular = torch.zeros_like(diffuse)
        if visible == None: # Simulate
            visible = self.estimate_visibility_test(model, points, **kwargs)

            rays_d = self.direction.unsqueeze(0).repeat(points.size(0), 1)
            label = torch.max(semantic, dim=-1)[1]
            vehicle_mask = (label == VEHICLE_LABEL)
            half_d = F.normalize(-view_d + rays_d, dim=-1)
            normal = F.normalize(normal, dim=-1)
            specular = torch.sum(half_d * normal, dim=-1)
            specular = vehicle_mask * self.specular_strength * (torch.clip(specular, 0.0, 1.0) ** self.shininess)
        shading = ((diffuse + specular)*visible).unsqueeze(1) + self.ambient.unsqueeze(0)
        rgb = albedo * shading

        label = torch.max(semantic, dim=-1)[1]
        sky_mask = (label == SKY_LABEL)
        rgb_sky = self(view_d)
        rgb[sky_mask] = rgb_sky[sky_mask] * self.sky_shading.unsqueeze(0)
        rgb = torch.clip(rgb, 0.0, 1.0)
        return rgb, visible

    def get_tracing_rays(
        self,
        model,
        points
    ):
        n_pts = points.size(0)
        light_d = self.direction.unsqueeze(0).repeat(n_pts, 1)
        light_dist = (self.sky_height - points @ self.up) / (light_d @ self.up)

        NEAR_DISTANCE_T = self.near_dist_T
        _, hits_t, _ = RayAABBIntersector.apply(points, light_d, model.center, model.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE_T), 0, 0] = NEAR_DISTANCE_T
        hits_t = hits_t[:,0,:] 
        return points, light_d, light_dist, hits_t
    
    def estimate_visibility_test(
        self,
        model,
        points,
        **kwargs
    ):
        rays_o, rays_d, light_dist, hits_t = self.get_tracing_rays(model, points)
        visibility = sample_visibility_test(model, rays_o, rays_d, hits_t, light_dist, **kwargs)
        return visibility

    def estimate_visibility_train(
        self,
        model,
        points,
        **kwargs
    ):
        # print('points:', points.size())
        rays_o, rays_d, light_dist, hits_t = self.get_tracing_rays(model, points)

        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        with torch.no_grad():
            rays_a, xyzs, dirs, deltas, ts, total_samples = \
                RayMarcher.apply(
                    rays_o, rays_d, hits_t, model.density_bitfield,
                    model.cascades, model.scale,
                    exp_step_factor, model.grid_size, MAX_SAMPLES)
        
        sigmas = model.density(xyzs)
        visibility, ws = VolumeRendererVisibility.apply(
            sigmas, deltas, ts, rays_a, light_dist, kwargs.get('T_threshold', 1e-4)
        )
        return visibility

if __name__ == '__main__':
    model = SunSkyModel()
