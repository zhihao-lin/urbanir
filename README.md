# UrbanIR: Large-Scale Urban Scene Inverse Rendering from a Single Video

### [Project Page](https://urbaninverserendering.github.io/) | [Paper](https://arxiv.org/abs/2306.09349) | [Data](https://uofi.box.com/s/c6ocdrqktrbah661cmvw9njcfqu24ric) | [Checkpoints](https://uofi.box.com/s/4e4ud4dwgwfqwoytz66emywauyrneqxz)

[Zhi-Hao Lin<sup>1</sup>](https://zhihao-lin.github.io/), [Bohan Liu<sup>1</sup>](https://www.linkedin.com/in/bohanliu524/?locale=en_US), [Yi-Ting Chen<sup>2</sup>](https://jamie725.github.io/website/), Kuan-Sheng Chen<sup>1</sup>, [David Forsyth<sup>1</sup>](http://luthuli.cs.uiuc.edu/~daf/), [Jia-Bin Huang<sup>2</sup>](https://jbhuang0604.github.io/), [Anand Bhattad<sup>1</sup>](https://anandbhattad.github.io/), [Shenlong Wang<sup>1</sup>](https://shenlong.web.illinois.edu/)

<sup>1</sup>University of Illinois at Urbana-Champaign, <sup>2</sup>University of Maryland, College Park

![teaser](docs/images/teaser.jpg)

## 🔦 Prerequisites
The code has been tested on:
- **OS**: Ubuntu 22.04.4 LTS
- **GPU**: NVIDIA GeForce RTX 4090, NVIDIA RTX A6000
- **Driver Version**: 535, 545
- **CUDA Version**: 12.2, 12.3
- **nvcc**: 11.7

## 🔦 Installation
- Create Conda environment:
```
conda create -n urbanir -y python=3.9
conda activate urbanir
```
- Install python packages:
```
pip install -r requirements.txt
```
- Install [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter):
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
- Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
```
Then use your favorite editor to edit `tiny-cuda-nn/include/tiny-cuda-nn/common.h` and set `TCNN_HALF_PRECISION` to `0` (see [NVlabs/tiny-cuda-nn#51](https://github.com/NVlabs/tiny-cuda-nn/issues/51) for details)
```
cd tiny-cuda-nn/bindings/torch
python setup.py install
```
- Compile CUDA extension of this project
```
pip install models/csrc/
```

## 🔦 Dataset and Checkpoints
- Please download [datasets](https://uofi.box.com/s/c6ocdrqktrbah661cmvw9njcfqu24ric) and put under `data/` folder.
- Please download [checkpoints](https://uofi.box.com/s/4e4ud4dwgwfqwoytz66emywauyrneqxz) and put under `ckpts` folder.
- Currently the data and checkpoints of [Kitti360](https://www.cvlibs.net/datasets/kitti-360/) and [Waymo Open Dataset](https://waymo.com/open/) are available.

## 🔦 Training
- The training process is tracked and visualized with [wandb](https://github.com/wandb/wandb), you could set up by `$ wandb login`
- The training script examples are in `scripts/train.sh`, which are in the following format:
```
python train.py --config [path_to_config]
```
- The checkpoint is saved to `ckpts/[dataset]/[exp_name]/*.ckpt`
- The validation images are saved to `results/[dataset]/[exp_name]/val`

## 🔦 Rendering
- The rendering script examples are in `scripts/render.sh`, which are in the following format:
```
python render.py --config [path_to_config]
```
- The rendered images are saved to `results/[dataset]/[exp_name]/frames`, and videos are saved to `results/[dataset]/[exp_name]/*.mp4`
- If the training is not complete and `.../last_slim.ckpt` is not available, you can either specify path with `--ckpt_load [path_to_ckpt]` or convert with `utility/slim_ckpt.py`.

## 🔦 Relighting
- The relighting script examples are in `scripts/relight.sh`, which are in the following format:
```
python render.py --config [path_to_config] \
     --light [path_to_light_config] --relight [effect_name]
```
- The rendered images and videos are saved to `results/[dataset]/[exp_name]/[effect_name]`

## 🔦 Configuration
- All the parameters are listed in the `opt.py`, and can be added after training/rendering/relighting script with `--param_name value`.
- The scene-specific parameters are listed in `configs/[dataset]/[scene].txt`.
- The lighting parameters are listed in `configs/light/[scene].txt`, and different relighting effects can be produced by changing the configuration.

## 🔦 Customized Data
- The camera poses are estimated with [NeRFstudio pipeline](https://docs.nerf.studio/quickstart/custom_dataset.html) (`transforms.json`).
- The depth is estimated with [MiDaS](https://github.com/isl-org/MiDaS).
- The normal is estimated with [OmniData](https://github.com/EPFL-VILAB/omnidata), and [authors' fork](https://github.com/zhihao-lin/omnidata) can handle image resolution in a more flexible manner.
- The shadow mask is estimated with [MTMT](https://github.com/eraserNut/MTMT), and [authors' fork](https://github.com/zhihao-lin/MTMT) provides example script.
- The semantic map is estimated with [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), and you could put `mmseg/run.py` in the root folder of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and run.


## 🔦 Citation
If you find this paper and repository useful for your research, please consider citing: 
```bibtex
@article{lin2023urbanir,
  title={Urbanir: Large-scale urban scene inverse rendering from a single video},
  author={Lin, Zhi-Hao and Liu, Bohan and Chen, Yi-Ting and Forsyth, David and Huang, Jia-Bin and Bhattad, Anand and Wang, Shenlong},
  journal={arXiv preprint arXiv:2306.09349},
  year={2023}
}
```