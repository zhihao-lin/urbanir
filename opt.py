import argparse
import configargparse

def get_opts():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # common args for all datasets
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nerf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 
                                 'tnt', 'kitti', 'mega', 'highbay', 'waymo'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    parser.add_argument('--anti_aliasing_factor', type=float, default=1.0,
                        help='Render larger images then downsample')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    parser.add_argument('--embed_a', action='store_true', default=False,
                        help='whether to use appearance embeddings')
    parser.add_argument('--embed_a_len', type=int, default=4,
                        help='the length of the appearance embeddings')
    parser.add_argument('--embed_msk', action='store_true', default=False,
                        help='whether to use sigma embeddings')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='total number of semantic classes')

    # for kitti 360 dataset
    parser.add_argument('--kitti_seq', type=int, default=0, 
                        help='scene sequence index')
    parser.add_argument('--kitti_start', type=int, default=1538,
                        help='starting frame index')
    parser.add_argument('--kitti_end', type=int, default=1601,
                        help='ending frame index')
    parser.add_argument('--kitti_test_id', type=int, nargs='+', default=[],
                        help='frames for testing')

    # for waymo dataset
    parser.add_argument('--waymo_start', type=int, default=0)
    parser.add_argument('--waymo_end', type=int, default=100)
    parser.add_argument('--waymo_sun', type=float, nargs='+',
                        default=[-0.07508344, -0.05114832, -0.00162191],
                        help='sun light direction')

    # training options
    parser.add_argument('--nvs', action='store_true', default=False,
                        help='Evaluate NVS and exclude testing frames in training')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--normal_epochs', type=int, default=20,
                        help='number of training epochs for normal distillation')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument('--density_threshold', type=float, default=1e-2,
                        help='threshold for updating density grid')
    parser.add_argument('--depth_mono', action='store_true', default=False,
                        help='use 2D depth prediction as supervision')
    parser.add_argument('--normal_mono', action='store_true', default=False,
                        help='use 2D normal prediction as supervision')
    parser.add_argument('--normal_ref', action='store_true', default=False,
                        help='use density gradient as normal supervision (Ref-NeRF)')
    parser.add_argument('--visibility', action='store_true', default=False,
                        help='supervise sivibility with shadow masks (MTMT)')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics (experimental)')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
    parser.add_argument('--warmup_steps', type=int, default=256)
    parser.add_argument('--update_interval',type=int, default=16)

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--render_traj', action='store_true', default=False,
                        help='render video on a trajectory')
    parser.add_argument('--render_train', action='store_true', default=False,
                        help='interpolate among training views to get camera trajectory')

    # Loss weight 
    parser.add_argument('--l_opa', type=float, default=2e-4)
    parser.add_argument('--l_distortion', type=float, default=1e-3)
    parser.add_argument('--l_depth_mono', type=float, default=0.0)
    parser.add_argument('--l_normal_mono', type=float, default=1e-2)
    parser.add_argument('--l_normal_ref_rp', type=float, default=1e-3)
    parser.add_argument('--l_normal_ref_ro', type=float, default=1e-3)
    parser.add_argument('--l_sky', type=float, default=1e-1)
    parser.add_argument('--l_semantic', type=float, default=4e-2)
    parser.add_argument('--l_sparsity', type=float, default=1e-4)
    parser.add_argument('--l_visibility', type=float, default=1e-3)
    parser.add_argument('--l_visibility_T', type=float, default=1e-4)
    parser.add_argument('--l_deshadow', type=float, default=0.0)
    parser.add_argument('--l_albedo', type=float, default=1e-3)
    parser.add_argument('--l_ambient', type=float, default=0.0)
    parser.add_argument('--l_init_shading_prior', type=float, default=1.0)

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_load', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--ckpt_save', type=str, default='checkpoint.ckpt',
                        help='pretrained checkpoint to save (including optimizers, etc)')
    
    # render
    parser.add_argument('--render_rgb', action='store_true', default=False,
                        help='render rgb series')
    parser.add_argument('--render_albedo', action='store_true', default=False,
                        help='render albedo series')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='render depth series')
    parser.add_argument('--render_normal', action='store_true', default=False,
                        help='render normal series')
    parser.add_argument('--render_semantic', action='store_true', default=False,
                        help='render semantic segmentation series')
    parser.add_argument('--render_visibility', action='store_true', default=False,
                        help='render visibility from light source')
    parser.add_argument('--render_shading', action='store_true', default=False,
                        help='render shading intensity from light source')
    parser.add_argument('--normal_composite', action='store_true', default=False,
                        help='render normal+rgb composition series')
    parser.add_argument('--chunk_size', type=int, default=131072, 
                        help='Divide image into chunks for rendering')
    # Relighting
    parser.add_argument('--sky_height', type=float, default=0.8)
    parser.add_argument('--near_dist_T', type=float, default=0.01)
    parser.add_argument('--relight_idx', type=str, default='',
                        help='relighting name')
    parser.add_argument('--light_config', type=str, default='configs/light/1538-1601.yaml',
                        help='path to lighting config')
    parser.add_argument('--add_flares', action='store_true', default=False,
                        help='add flares at night')
    parser.add_argument('--video_len', type=int, default=400)

    # insertion 
    parser.add_argument('--insert_idx', type=str, default='',
                        help='insertion name w/ blender output')
    parser.add_argument('--insert_frame', type=int, default=0,
                        help='insertion frame index')
    return parser.parse_args()
