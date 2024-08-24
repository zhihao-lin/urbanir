import os
import numpy as np
import cv2
from mmseg.apis import inference_model, init_model
from tqdm import tqdm
from argparse import ArgumentParser

# cityscape label: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
def save_pgm(result, out_file):
    pred = result.pred_sem_seg.data
    pred = pred.squeeze().detach().cpu().numpy()
    
    seg_res = np.zeros(pred.shape)
    seg_res[:] = 9
    seg_res[pred==0]  = 0 # road
    seg_res[pred==1]  = 1 # sidewalk/flat
    seg_res[np.logical_and(pred>=2, pred<=4)] = 2 # construction
    seg_res[np.logical_or(pred==6, pred==7)] = 3 # object
    seg_res[np.logical_or(pred==8, pred==9)] = 4 # nature
    seg_res[pred==10] = 5 # sky
    seg_res[np.logical_or(pred==11, pred==12)] = 6 # human
    seg_res[pred>=13] = 7 # vehicle
    seg_res[pred == 5] = 8 # pole
    cv2.imwrite(out_file, seg_res)

parser = ArgumentParser()
parser.add_argument('--dir_rgb')
parser.add_argument('--dir_sem')
args = parser.parse_args()

config_file = 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
checkpoint_file = 'ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

rgb_path = args.dir_rgb # PATH to images' folder
semantic_path = args.dir_sem  # create folder for inference results
os.makedirs(semantic_path, exist_ok=True)
for file in tqdm(sorted(os.listdir(rgb_path))):
     img = os.path.join(rgb_path, file) # or img = mmcv.imread(img), which will only load it once
     result = inference_model(model, img)
     save_pgm(result, os.path.join(semantic_path, file.replace('.png', '.pgm'))) # save as .pgm file
