import torch 
from einops import rearrange
import vren
from models.const import MAX_SAMPLES

def sample_visibility_test(
    model, 
    rays_o,
    rays_d,
    hits_t,
    light_dist,
    # Other parameters
    **kwargs
):
    N_rays = len(rays_o) 
    device = rays_o.device
    exp_step_factor = kwargs.get('exp_step_factor', 0.)

    visibility = torch.ones(N_rays, device=device)
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4
    samples = 0
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
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        ## Shapes
        # xyzs: (N_alive*N_samples, 3)
        # dirs: (N_alive*N_samples, 3)
        # deltas: (N_alive, N_samples) intervals between samples (with previous ones)
        # ts: (N_alive, N_samples) ray length for each samples
        # N_eff_samples: (N_alive) #samples along each ray <= N_smaples

        sigmas = torch.zeros(len(xyzs), device=device)
        _sigmas = model.density(xyzs[valid_mask], return_feat=False, grad=False, grad_feat=False)

        sigmas[valid_mask] = _sigmas.detach().float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)

        vren.visibility_test_fw(
            sigmas, deltas, ts, alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, light_dist, visibility)
        alive_indices = alive_indices[alive_indices>=0]

    return visibility