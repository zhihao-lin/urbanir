import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import vren
import math
from models.const import SKY_LABEL, GROUND_LABEL, SIDEWALK_LABEL
from utils import get_mask_from_label

def compute_scale_and_shift(prediction, target):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(prediction * prediction)
    a_01 = torch.sum(prediction)
    ones = torch.ones_like(prediction)
    a_11 = torch.sum(ones)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(prediction * target)
    b_1 = torch.sum(target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    # x_0 = torch.zeros_like(b_0)
    # x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0 = torch.FloatTensor(0).cuda()
        x_1 = torch.FloatTensor(0).cuda()

    return x_0, x_1

class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)
    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None
    
class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))

class NeRFLoss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.lambda_opa = hparams.l_opa
        self.lambda_distortion = hparams.l_distortion # default
        # self.lambda_distortion = 1e-4 # for meganerf
        self.lambda_depth_mono = hparams.l_depth_mono
        self.lambda_normal_mono = hparams.l_normal_mono
        self.lambda_normal_ref_rp = hparams.l_normal_ref_rp
        self.lambda_normal_ref_ro = hparams.l_normal_ref_ro
        self.lambda_sky = hparams.l_sky
        self.lambda_semantic = hparams.l_semantic
        self.lambda_sparsity = hparams.l_sparsity
        self.lambda_visibility = hparams.l_visibility
        self.lambda_visibility_T = hparams.l_visibility_T
        self.lambda_deshadow = hparams.l_deshadow
        self.lambda_albedo = hparams.l_albedo 
        self.reg_ambient = hparams.l_ambient
        self.lambda_rgb = 1
        self.lambda_rgb_shading = 1

        self.Annealing = ExponentialAnnealingWeight(max = 1, min = 6e-2, k = 1e-3)
        
        self.sem_ce = nn.CrossEntropyLoss(ignore_index=256)
        self.vis_ce = nn.CrossEntropyLoss()

    def forward(self, results, target, **kwargs):
        d = {}
        
        d['rgb'] = self.lambda_rgb * (results['rgb']-target['rgb'])**2
        if self.lambda_rgb_shading > 0:
            d['rgb_shading'] = self.lambda_rgb_shading * (results['rgb_shading']-target['rgb'])**2
    
        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opa*(-o*torch.log(o))
        
        # decay with depth 
        depth_pred = results['depth']
        dist_decay = torch.exp(-depth_pred.detach())

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
            DistortionLoss.apply(results['ws'], results['deltas'],
                                    results['ts'], results['rays_a'])

        if kwargs.get('normal_ref', False):# for ref-nerf model
            d['normal_ref_rp'] = self.lambda_normal_ref_rp * results['Rp'] 
            d['normal_ref_ro'] = self.lambda_normal_ref_ro * results['Ro'] 

        if kwargs.get('normal_mono', False):
            normal_pred = F.normalize(results['normal_pred'], dim=-1)
            normal_gt   = F.normalize(target['normal'], dim=-1)
            l1_loss = torch.abs(normal_pred - normal_gt) #(n, 3)
            cos_loss = -(normal_pred * normal_gt) #(n, 3)
            d['normal_mono'] = self.lambda_normal_mono * (l1_loss + 0.1 * cos_loss)

        if kwargs.get('semantic', False):
            d['CELoss'] = self.lambda_semantic*self.sem_ce(results['semantic'], target['label'])
            sky_mask, _ = get_mask_from_label(target['label'], [SKY_LABEL])
            d['sky_depth'] = self.lambda_sky*sky_mask*torch.exp(-results['depth'])

        if kwargs.get('visibility', False):
            shadow_gt = target['shadow']
            visibility_gt = torch.stack([shadow_gt, 1-shadow_gt], dim=-1)
            visibility_pred = results['visibility']
            d['vis'] = self.lambda_visibility * self.vis_ce(visibility_pred, visibility_gt) 
            
            # transmittance to sun
            visibility_T = results['visibility_T']
            vis_T = torch.stack([1 - visibility_T, visibility_T], dim=-1)
            d['vis_T'] = self.lambda_visibility_T * self.vis_ce(vis_T, visibility_pred)

        if self.lambda_albedo > 0:            
            # enforce homogeneous albedo on road & sidewalk
            albedo = results['albedo']
            visible = ((1 - shadow_gt) + 1e-4).unsqueeze(-1)
            label = target['label']
            seg_idxs, inv_idxs = torch.unique(label, return_inverse=True)
            weight_seg = torch.zeros(len(seg_idxs), 1, device=seg_idxs.device)
            weight_seg = torch_scatter.scatter(visible, inv_idxs, 0, weight_seg, reduce='sum')

            albedo_cls = torch.zeros(len(seg_idxs), 3, device=seg_idxs.device)
            albedo_cls = torch_scatter.scatter(
                albedo.detach()*visible, inv_idxs, 0, albedo_cls, reduce='sum')
            albedo_cls_mean = albedo_cls / weight_seg
            albedo_tgt = albedo_cls_mean[inv_idxs]
            mask, _ = get_mask_from_label(target['label'], [GROUND_LABEL, SIDEWALK_LABEL])
            d['albedo'] = self.lambda_albedo * mask.unsqueeze(-1) * (albedo - albedo_tgt)**2


        if kwargs.get('depth_mono', False): # for kitti360 dataset
            depth_pred = results['depth']
            depth_tgt = target['depth']
            mask = target['label']!=SKY_LABEL
            weight = torch.zeros_like(depth_tgt)
            weight[mask] = 1.
            scale, shift = compute_scale_and_shift(depth_pred[mask].detach(), depth_tgt[mask])
            l2 = (scale * depth_pred + shift - depth_tgt)**2
            d['depth_mono'] = self.lambda_depth_mono * dist_decay * l2
        
        if self.reg_ambient > 0:
            amb = results['ambient']
            loss_amb = torch.mean(amb**2)
            d['ambient'] = self.reg_ambient * loss_amb
        
        return d

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit