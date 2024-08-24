import torch 
from utils import slim_ckpt, load_ckpt
import argparse 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    ckpt = slim_ckpt(args.input, save_poses=False)
    torch.save(ckpt, args.output)
    print('Save checkpoints to {}'.format(args.output))

if __name__ == '__main__':
    main()