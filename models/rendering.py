import numpy as np
from logging.config import valid_ident
import torch
import torch.nn.functional as F
from models.custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren
from models.volume_render import volume_render
from models.shading import SpotLight, SunLight, surface_points
from models.const import NEAR_DISTANCE, MAX_SAMPLES, SKY_LABEL

def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results

@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Input:
        rays_o: [h*w, 3] rays origin
        rays_d: [h*w, 3] rays direction

    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    hits_t = hits_t[:,0,:]
    # Perform volume rendering
    opacity, depth, rgb, albedo, normal_pred, normal_raw, semantic, visibility, total_samples = \
        volume_render(model, rays_o, rays_d, hits_t, **kwargs)
    normal_raw  = F.normalize(normal_raw, dim=-1)
    normal_pred = F.normalize(normal_pred, dim=-1)

    surface = surface_points(rays_o, rays_d, depth)
    light = kwargs.get('light', None)
    if not kwargs.get('relight', False):
        visible = 1 - visibility[..., 0]
        rgb_shading, _ = light.shading_model(
            model, surface, albedo, normal_pred, semantic, 
            rays_d, visible=visible, **kwargs)
        visibility_T = light.estimate_visibility_test(model, surface, **kwargs)
    else:
        rgb_shading, visible = light.shading_model(
            model, surface, albedo, normal_pred, semantic, 
            rays_d, **kwargs)
        visibility_T = visible

    results = {}
    results['opacity'] = opacity # (h*w)
    results['depth'] = depth # (h*w)
    results['albedo'] = albedo # (h*w, 3)
    results['rgb'] = rgb # (h*w, 3)
    results['rgb_shading'] = rgb_shading # (h*w, 3)
    results['normal_pred'] = normal_pred
    results['normal_raw'] = normal_raw
    results['semantic'] = torch.argmax(semantic, dim=-1)
    results['total_samples'] = total_samples # total samples for all rays
    results['visibility'] = visible
    results['visibility_T'] = visibility_T

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    light = kwargs.get('light', None)
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        rays_a, xyzs, dirs, deltas, ts, total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['rays_a'] = rays_a
    results['deltas'] = deltas
    results['ts'] = ts
    results['total_samples'] = total_samples
    
    repeat_keys = ['embedding_a']
    for k, v in kwargs.items():
        if k in repeat_keys:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    sigmas, rgbs, albedos, normals_raw, normals_pred, sems, viss = model(xyzs, dirs, **kwargs)
    results['sigma'] = sigmas
    results['xyzs'] = xyzs
    
    # xyz_sparse = (torch.rand_like(xyzs) - 0.5) * 2 * model.scale 
    # sigma_sparse = model.density(xyz_sparse)
    # results['sigma_sparse'] = sigma_sparse

    vr_samples, opacity, depth, rgb, albedo, normal_pred, semantic, visibility, ws = \
        VolumeRenderer.apply(sigmas.contiguous(), rgbs.contiguous(), albedos.contiguous(), normals_pred.contiguous(), 
                                sems.contiguous(), viss.contiguous(), deltas, ts,
                                rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('num_classes', 7))
    
    # normal_pred = F.normalize(normal_pred, dim=-1)
    diffuse = torch.sum(normal_pred * light.direction.unsqueeze(0), dim=-1) 
    diffuse = torch.clip(diffuse, 0.0, 1.0)
    
    visible = 1 - visibility[..., 0]
    shading = (diffuse * visible).unsqueeze(1) + light.ambient.unsqueeze(0)
    rgb_shading = albedo * shading

    shading_deshadow = diffuse.unsqueeze(1) + light.ambient.unsqueeze(0)
    rgb_deshadow = albedo * shading_deshadow

    # add random background
    rand_bg = torch.rand(3, device=rgb.device) * (1 - opacity)[:, None]
    rgb_shading  += rand_bg
    rgb_deshadow += rand_bg
    
    # sky appearance
    label = kwargs.get('label', None)
    # label = torch.max(semantic, dim=-1)[1]
    sky_mask = (label == SKY_LABEL)
    rgb_sky = light(rays_d)
    rgb_shading[sky_mask] = rgb_sky[sky_mask]
    rgb_deshadow[sky_mask] = rgb_sky[sky_mask]

    results['opacity'] = opacity
    results['depth'] = depth
    results['albedo'] = albedo
    results['normal_pred'] = normal_pred
    results['semantic'] = semantic
    results['visibility'] = visibility
    results['ws'] = ws
    results['rgb'] = rgb 
    results['rgb_shading'] = rgb_shading
    results['rgb_deshadow'] = rgb_deshadow
    results['ambient'] = light.ambient
    results['light_dir'] = light.direction

    # Normal loss
    normals_diff = (normals_raw - normals_pred)**2
    dirs = F.normalize(dirs, p=2, dim=-1, eps=1e-6)
    normals_ori = torch.clamp(torch.sum(normals_raw*dirs, dim=-1), min=0.)**2 # don't keep dim!
    
    results['Ro'], results['Rp'] = \
        RefLoss.apply(sigmas.detach().contiguous(), normals_diff.contiguous(), normals_ori.contiguous(), results['deltas'], results['ts'],
                            rays_a, kwargs.get('T_threshold', 1e-4))

    # Transmittance loss 
    # mask_t = light.get_mask_with_w(ws)
    # if mask_t.sum() == 0:
    #     loss_t = torch.zeros_like(mask_t, device=ws.device)
    # else:
    #     xyzs_t = xyzs[mask_t]
    #     viss_t = light.estimate_visibility_train(model, xyzs_t, **kwargs) # (t,)
    #     viss_t = torch.stack([1-viss_t, viss_t], dim=-1)
    #     viss_l = viss[mask_t] # (t, 2)
    #     ws_t = ws[mask_t]
    #     loss_t = weighted_cross_entropy(viss_t, viss_l, ws_t)
    # results['loss_transmittance'] = loss_t
    surface = rays_o + rays_d * depth.unsqueeze(-1)
    visibility_T = light.estimate_visibility_train(model, surface, **kwargs)
    results['visibility_T'] = visibility_T

    return results

def weighted_cross_entropy(x, y, weight):
    '''
    x: (n, d) prediction
    y: (n, d) target distribution
    w: (n, )
    '''
    log_x = torch.log(F.softmax(x, dim=1))
    ce = -y * log_x * weight.unsqueeze(1)
    ce = torch.mean(torch.sum(ce, dim=1))
    return ce