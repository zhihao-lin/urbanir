import torch
from einops import rearrange
import vren
from models.const import MAX_SAMPLES

def volume_render(
    model, 
    rays_o,
    rays_d,
    hits_t,
    # Other parameters
    **kwargs
):
    N_rays = len(rays_o) 
    device = rays_o.device
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    classes = kwargs.get('num_classes', 10)

    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    albedo = torch.zeros(N_rays, 3, device=device)
    normal_pred = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)
    semantic = torch.zeros(N_rays, classes, device=device)
    visibility = torch.zeros(N_rays, 2, device=device)

    samples = 0
    total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4
    
    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t, alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        # valid_mask = torch.all(dirs==0, dim=1) # NOTE: for unit test only
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        ## Shapes
        # xyzs: (N_alive*N_samples, 3)
        # dirs: (N_alive*N_samples, 3)
        # deltas: (N_alive, N_samples) intervals between samples (with previous ones)
        # ts: (N_alive, N_samples) ray length for each samples
        # N_eff_samples: (N_alive) #samples along each ray <= N_smaples

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        albedos = torch.zeros(len(xyzs), 3, device=device)
        normals_pred = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        sems = torch.zeros(len(xyzs), classes, device=device)
        viss = torch.zeros(len(xyzs), 2, device=device)

        _sigmas, _rgbs, _albedos, _normals_pred, _normals_raw, _sems, _viss = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)

        sigmas[valid_mask] = _sigmas.detach().float()
        rgbs[valid_mask] = _rgbs.detach().float()
        albedos[valid_mask] = _albedos.float()
        normals_pred[valid_mask] = _normals_pred.float()
        normals_raw[valid_mask] = _normals_raw.float()
        sems[valid_mask] = _sems.float()
        viss[valid_mask] = _viss.float()
        
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        albedos = rearrange(albedos, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        viss = rearrange(viss, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, rgbs, albedos, normals_pred, normals_raw, sems, viss, deltas, ts,
            hits_t, alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, albedo, normal_pred, normal_raw, semantic, visibility)
        alive_indices = alive_indices[alive_indices>=0]

    return opacity, depth, rgb, albedo, normal_pred, normal_raw, semantic, visibility, total_samples

def test():
    # NOTE: need to change valid_mask for running vren.composite_test_fw
    from models.networks import NGP
    from models.custom_functions import RayAABBIntersector
    from models.const import NEAR_DISTANCE
    import time
    # test the density function
    embed_a_len = 12
    num_classes = 7
    sample_n = 1000
    model = NGP(
        scale=4, 
        rgb_act='Sigmoid', 
        embed_a=True, 
        embed_a_len=embed_a_len, 
        classes=num_classes
    ).cuda()

    emb = torch.randn(1, embed_a_len).cuda()
    kwargs = {
        'embedding_a': emb,
        'num_classes': num_classes,
    }
    
    rays_o = torch.randn(sample_n, 3).cuda()
    rays_d = torch.randn(sample_n, 3).cuda()
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    start = time.time()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
    hits_t = hits_t[:,0,:]
    print('Finished ray marching: {:.5f} s'.format(time.time()-start))

    start = time.time()
    opacity, depth, rgb, albedo, normal_pred, normal_raw, semantic, visibility, total_samples = \
        volume_render(model, rays_o, rays_d, hits_t, **kwargs)
    print('Finished volume rendering: {:.5f} s'.format(time.time()-start))
    print('RGB:', rgb.size(), rgb.sum())
    print('Albedo:', albedo.size(), albedo.sum())

if __name__ == '__main__':
    test()