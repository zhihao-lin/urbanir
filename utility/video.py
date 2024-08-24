import os 
import cv2 
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
from PIL import Image
import numpy as np
import argparse

def extract_frames():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', help='path to video')
    parser.add_argument('-outdir', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    vidcap = cv2.VideoCapture(args.video)
    i = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        path = os.path.join(args.outdir, '{:0>5d}.png'.format(i))
        cv2.imwrite(path, image)
        i += 1

def read_video_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
    if len(frames) == 0:
        print('ERROR: {} does not exist'.format(video_path))
    return frames

def is_image_name(name):
    valid_names = ['.jpg', '.png', '.JPG', '.PNG']
    for v in valid_names:
        if name.endswith(v):
            return True
    return False

def generate_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-out')
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()

    imgs = sorted([os.path.join(args.dir, img) for img in os.listdir(args.dir) if is_image_name(img)])
    imgs = [np.array(Image.open(img)) for img in imgs]

    imageio.mimsave(args.out, imgs, fps=args.fps, macro_block_size=1)

    # imgs = [Image.open(img) for img in imgs]
    # imgs += imgs[::-1]
    # imgs[0].save(args.out, format='GIF', append_images=imgs,
    #      save_all=True, duration=10, loop=0)

def add_text():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_in')
    parser.add_argument('-video_out')
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()

    background_color = (0, 0, 0)  # Red color in BGR format
    text = "Relight"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  # White color in BGR format
    thickness = 4

    frames_in = read_video_frames(args.video_in)
    frames_out = []
    for frame in frames_in:
        parser = argparse.ArgumentParser()
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = 10, 10 # Position of the text
        x2, y2 = x + text_size[0] + 10, y + text_size[1] + 10

        cv2.rectangle(frame, (x, y), (x2, y2), background_color, -1)
        cv2.putText(frame, text, (x + 5, y + 50), font, font_scale, font_color, thickness)
        frames_out.append(frame)

    imageio.mimsave(args.video_out,
                    frames_out,
                    fps=args.fps, macro_block_size=1)



def merge_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-first')
    parser.add_argument('-second')
    parser.add_argument('-out')
    parser.add_argument('-axis', type=int, default=0)
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()

    frames_first  = read_video_frames(args.first)
    frames_second = read_video_frames(args.second)
    frames_out   = []

    frame_len = min(len(frames_first), len(frames_second))
    for i in range(frame_len):
        left  = frames_first[i]
        right = frames_second[i]
        out = np.concatenate([left, right], axis=args.axis)
        frames_out.append(out)

    imageio.mimsave(args.out,
                    frames_out,
                    fps=args.fps, macro_block_size=1)
    print('Export video:', args.out)

def test():
    root = 'results/waymo/240414_waymo_seq0_base'
    dir_frames = os.path.join(root, 'frames')
    post_fixes = ['albedo.png', 'depth.png', 'normal-raw.png', 'normal.png', 'rgb.png', 'shading.png', 'vis-T.png', 'vis.png']
    videos = ['render_albedo.mp4', 'render_depth.mp4', 'render_normal_raw.mp4', 'render_normal.mp4', 'render_rgb.mp4', 'render_shading.mp4', 'render_visibility_T.mp4', 'render_visibility.mp4']

    for i in range(len(post_fixes)):
        post = post_fixes[i]
        imgs = sorted([os.path.join(dir_frames, name) for name in os.listdir(dir_frames) if name.endswith(post)])
        imgs = [np.array(Image.open(img)) for img in imgs]
        print(post)
        print('imgs len:', len(imgs))
        save_path = os.path.join(root, videos[i])
        imageio.mimsave(save_path, imgs, fps=30, macro_block_size=1)
        print('save to ', save_path)

if __name__ == '__main__':
    # extract_frames()
    # generate_video()
    merge_video()
    # add_text()
    # test()