# read *.mp4 files and calculate the mean and variance of each frame

import os, imageio, decord, json
import cv2
import numpy as np
from glob import glob
from einops import reduce
from tqdm import tqdm

class RunningMeanStd(object):
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')  # μ
        self.var = np.ones(shape, 'float64')  # σ^2
        self.n = 0  # n

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_n = x.shape[0]

        delta = batch_mean - self.mean
        total_n = self.n + batch_n

        # 均值
        self.mean = self.mean + delta * batch_n / total_n

        # 方差
        self.var = (self.var * self.n + batch_var * batch_n + np.square(delta) * self.n * batch_n / total_n) / total_n

        # 数量
        self.n = total_n

    def get(self, n_precision=6):
        return self.mean, np.sqrt(self.var)  # (μ,σ)  .round(n_precision)

def mp42np(mp4_file, way='decord'):
    if way == 'decord':
        return [im.asnumpy() for im in decord.VideoReader(mp4_file, num_threads=16)]
    elif way == 'ffmpeg':
        return [im for im in imageio.get_reader(mp4_file,  'ffmpeg')]


if __name__ == "__main__":
    domain = 'robothor'  # habitat
    # read files
    rgbvideo_files = glob(f'./offline-dataset/{domain}-dataset/900/train/FloorPlan*/rgb_video.mp4') #+ glob(f'./offline-dataset/habitat-dataset/900/*/rgb_video.mp4')
    out_name = f'./offline-dataset/mean_std.json'
    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            mean_std_dict = json.load(f)
    else:
        mean_std_dict = {}
    # mean_std_dict = {}    
    for video_file in tqdm(rgbvideo_files):
        scene_name = video_file.split('/')[-2]
        if scene_name not in mean_std_dict:
            running_mean_std = RunningMeanStd((480, 640, 3))
            images = np.stack(mp42np(video_file, way='decord')) / 255.
            running_mean_std.update(images)
            mu, sigma = running_mean_std.get()
            mu, sigma = reduce(mu, 'h w c -> c', 'mean').tolist(), reduce(sigma, 'h w c -> c', 'mean').tolist()
            mean_std_dict[scene_name] = [mu, sigma]
    mean_std_dict[domain] = [
        reduce(np.array([v[0] for v in mean_std_dict.values()]), 'n c -> c', 'mean').tolist(),  # mean
        reduce(np.array([v[1] for v in mean_std_dict.values()]), 'n c -> c', 'mean').tolist()   # std    
    ]
    for k, v in mean_std_dict.items():
        print(k, v)
    with open(out_name, 'w') as f:
        json.dump(mean_std_dict, f, indent=4)