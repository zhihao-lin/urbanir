import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']

def box_filter(image, r):
    '''
    Input
        image: (h, w)
        r: constant
    Return
        image: (h, w)
    '''
    image = image[None][None] #(1, 1, h, w)
    device = image.device
    filters = torch.ones(1, 1, 2*r+1, 2*r+1, device=device) / ((2*r+1)**2)
    image_pad = F.pad(image, (r, r, r, r), mode='reflect')
    image_out = F.conv2d(image_pad, filters)
    image_out = image_out[0, 0] #(h, w)
    return image_out

def guided_filter(image_p, image_i, r, eps=0.1):
    '''
    Input:
        image_p: input (h, w)
        image_i: guided (h, w)
        r: radius of filter window
        eps: regularization weight, higher->smooth
    '''
    mean_p  = box_filter(image_p, r)
    mean_i  = box_filter(image_i, r)
    corr_ip = box_filter(image_i*image_p, r)
    corr_ii = box_filter(image_i*image_i, r)

    var_i = corr_ii - mean_i * mean_i 
    cov_ip = corr_ip - mean_i * mean_p 
    a = cov_ip / (var_i + eps**2)
    b = mean_p - a * mean_i
    
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    image_out = mean_a * image_i + mean_b
    return image_out

def save_image(image, path):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

def convert_normal(normal, pose_c2w):
    R_w2c = pose_c2w[:3, :3].T
    normal_cam = normal @ R_w2c.T
    return normal_cam

def get_mask_from_label(label, ids):
    mask = torch.zeros_like(label).float()
    for i in ids:
        mask += (label == i)
    is_ids = mask > 0
    return mask, is_ids

class FrameEmbedding(nn.Module):
    def __init__(self, embed_a_len, poses, ckpt_path=None):
        super().__init__()
        self.poses = poses
        embedding_a = torch.nn.Embedding(len(poses), embed_a_len)
        if ckpt_path is not None:
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["model", "msk_model"])
        self.embedding_a = embedding_a
        # print('** embedding_a:', self.embedding_a.weight.shape)
    
    def forward(self, x, mode='index'):
        if mode == 'index':
            emb = self.sample_index(x)
        elif mode == 'first':
            emb = self.sample_index(0)
        elif mode == 'nearest':
            emb = self.sample_nearest(x)
        elif mode == 'mean':
            emb = self.sample_mean(x)
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        return emb
    
    def sample_index(self, index):
        if torch.is_tensor(index) == False:
            index = torch.tensor([index])
        emb = self.embedding_a(index)
        return emb

    def sample_nearest(self, pose):
        frames_t = self.poses[:, :3, -1]
        t = pose[:3, -1]
        dist = torch.sum((frames_t - t)**2, dim=1)
        argmin = torch.argmin(dist)
        index = torch.tensor([argmin])
        # print('index:', index)
        emb = self.embedding_a(index)
        return emb

    def sample_mean(self, pose):
        frames_t = self.poses[:, :3, -1]
        t = pose[:3, -1]
        dist = torch.sum((frames_t - t)**2, dim=1)
        _, indices = torch.topk(-dist, 2)
        # print('indices:', indices)
        embs = self.embedding_a(indices)
        emb  = torch.mean(embs, dim=0, keepdim=True)
        return emb

def test():
    poses = torch.randn(100, 3, 4)
    frame_emb = FrameEmbedding(8, poses)
    p = torch.randn(3, 4)
    emb = frame_emb(p, mode='mean')
    print('embedding shape:', emb.size())
    


if __name__ == '__main__':
    test()