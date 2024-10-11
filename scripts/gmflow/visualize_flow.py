import pdb
import shutil
import sys
from argparse import ArgumentParser

sys.path.append('core')
sys.path.append('scripts/gmflow')
import json
import os
import pdb
import random

# from rigid_transform import *
import time
from glob import glob

import cv2
import imageio
import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import zscore
from sklearn.cluster import KMeans
from torch import nn
from torch.nn.functional import max_pool2d
from torchvision.utils import draw_bounding_boxes, save_image
from tqdm import tqdm
from unimatch.flow_viz import flow_to_image
from unimatch.unimatch import UniMatch


def draw_flow(flow, save_file, tag=''):
    u,v = flow[..., 0], flow[..., 1]
    H, W = u.shape[:2]  
    y,x = np.mgrid[0:H, 0:W]

    # Normalize the flow vectors
    rad = np.sqrt(u ** 2 + v ** 2)
    rad_max = rad.max()
  
    u_n = u / (rad_max + torch.finfo(float).eps)
    v_n = v / (rad_max + torch.finfo(float).eps)

    # print(f'umax:{u.max()} | umin:{u.min()}')
    # print(f'vmax:{v.max()} | vmin:{v.min()}')
    # print(f'radmax:{rad.max()} | radmin:{rad.min()}')
    # print(f'unmax:{u_n.max()} | unmin:{u_n.min()}')
    # print(f'vnmax:{v_n.max()} | vnmin:{v_n.min()}')

    
    # 筛选出梯度大小排在前的向量 
    per = 99
    mask_per = rad > np.percentile(rad, per) 
    mask_freq = np.zeros((H, W))
    mask_freq[::4, ::4] = 1# regularily sample mask
    mask_new = np.logical_and(mask_per, mask_freq)


    # 应用筛选，准备可视化数据 
    x_filtered_new = x[mask_new] 
    y_filtered_new = y[mask_new] 
    u_filtered_new = u[mask_new] 
    v_filtered_new = v[mask_new] 
    gradient_filtered_new = rad[mask_new] 
    
    params = {
        "font.size": 14,  # 全局字号
        'font.family': 'STIXGeneral',  # 全局字体，微软雅黑(Microsoft YaHei)可显示中文
        "figure.subplot.wspace": 0.2,  # 图-子图-宽度百分比
        "figure.subplot.hspace": 0.4,  # 图-子图-高度百分比
        "axes.spines.right": True,  # 坐标系-右侧线
        "axes.spines.top": True,  # 坐标系-上侧线
        "axes.titlesize": 14,  # 坐标系-标题-字号
        "axes.labelsize": 14,  # 坐标系-标签-字号
        "legend.fontsize": 14,  # 图例-字号
        "xtick.labelsize": 12,  # 刻度-标签-字号
        "ytick.labelsize": 12,  # 刻度-标签-字号
        "xtick.direction": 'in',  # 刻度-方向
        "ytick.direction": 'in'  # 刻度-方向
    }
    plt.rcParams.update(params)
    plt.figure() 
    quiver_new = plt.quiver(x_filtered_new, y_filtered_new, u_filtered_new, v_filtered_new, gradient_filtered_new, scale=100, scale_units='inches',
                            alpha=0.5, 
                            width=0.05,  # Adjusts the width of the arrows
                            headwidth=3,  # Adjusts the width of the arrow heads
                            headlength=5,  # Adjusts the length of the arrow heads
                            headaxislength=4.5  # Adjusts the length of the head in the direction of the arrow
                            )
    colorbar = plt.colorbar(quiver_new) #, label='Gradient Magnitude'
    if not tag:
        tag = f'Visualization of Top {100-per}% Gradient Vectors'
    plt.title(tag) 

    # plt.xlim(0, W)
    # plt.ylim(0, H)
    plt.gca().invert_yaxis() 
    plt.xticks([])
    plt.yticks([])
    colorbar.ax.set_yticklabels([])

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    plt.savefig(save_file)
    plt.close()

def show(img, winname='image',bgr2rgb=True,save='',display=True):
    if img.ndim > 3:
        img = img[0]
    if not isinstance(img, np.ndarray):
        img = img.cpu().numpy()
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.resizeWindow(winname, 600, 600)
    if save:
        if not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))
        cv2.imwrite(save, img)
    if display:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, img)
        cv2.waitKey(0)

def np2mp4(nparray, save_file, actions=None, way='imageio', fps=10):
    '''
    nparray: list of (H, W, C) array
    '''
    assert len(nparray) > 0
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    H, W = nparray[0].shape[:2]
    if way == 'cv2':
        fourcc = cv2.VideoWriter_fourcc(*'MP42')  # 定义视频编解码器:mp4v
        video_writer = cv2.VideoWriter(save_file, fourcc, fps, (W, H))
        for frame in nparray:  # tqdm(,desc='Writing ndarray to mp4 using cv2'):
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_writer.write(frame)
        video_writer.release()
    if way == 'imageio':
        imageio.mimwrite(save_file, nparray, fps=fps, quality=10)  # quality: [0,10]
    if actions is not None:
        np.save(f"{os.path.splitext(save_file)[0]}.npy", actions)
    print(f'successfully collect {save_file}')

def mp42np(mp4_file):
    #cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    preprocess = lambda x: x#cv2.resize(x, expected_shape[::-1])
    return np.stack([preprocess(im) for im in imageio.get_reader(mp4_file,  'ffmpeg')])

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class dummy_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Unet()

def get_bbox_keypoints(img_size, label, r=4):
    w, h = img_size[2], img_size[1]
    x_mult, y_mult = w/100, h/100
    x0, y0, x1, y1 = label["x"]*x_mult, label["y"]*y_mult, (label["x"]+label["width"])*x_mult, (label["y"]+label["height"])*y_mult
    x_stride, y_stride = (x1-x0)/r, (y1-y0)/r
    kps = []
    for i in range(r):
        for j in range(r):
            x = x0 + x_stride * (i+0.5)
            y = y0 + y_stride * (j+0.5)
            kps.append((x, y))
    return kps, ((x0+x1)/2, (y0+y1)/2)

def sample_with_binear(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = int(kp[0]), int(kp[1])
    x1, y1 = x0+1, y0+1
    x, y = kp[0]-x0, kp[1]-y0
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def warp_kp_with_bilinear(flow, kp):
    max_x, max_y = flow.shape[1]-1, flow.shape[0]-1
    x0, y0 = int(kp[0]), int(kp[1])
    x1, y1 = x0+1, y0+1
    x, y = kp[0]-x0, kp[1]-y0
    flow_x0y0 = flow[y0, x0]
    flow_x1y0 = flow[y0, x1]
    flow_x0y1 = flow[y1, x0]
    flow_x1y1 = flow[y1, x1]
    flow_y0 = flow_x0y0 * (1-x) + flow_x1y0 * x
    flow_y1 = flow_x0y1 * (1-x) + flow_x1y1 * x
    flow = flow_y0 * (1-y) + flow_y1 * y
    new_kp = (np.clip(kp[0]+flow[0], 0, max_x-1), np.clip(kp[1]+flow[1], 0, max_y-1))
    return new_kp

def sample_with_binear_v2(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def warp_kp_with_bilinear_v2(flow, kp):
    max_x, max_y = flow.shape[1]-1, flow.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    flow_x0y0 = flow[y0, x0]
    flow_x1y0 = flow[y0, x1]
    flow_x0y1 = flow[y1, x0]
    flow_x1y1 = flow[y1, x1]
    flow_y0 = flow_x0y0 * (1-x) + flow_x1y0 * x
    flow_y1 = flow_x0y1 * (1-x) + flow_x1y1 * x
    flow = flow_y0 * (1-y) + flow_y1 * y
    new_kp = kp[0]+flow[0], kp[1]+flow[1]
    return new_kp    

def warp_points(flow, points):
    warped_points = []
    for kp in points:
        warped_points.append(warp_kp_with_bilinear(flow, kp))
    return np.array(warped_points)

def warp_points_v2(flow, points):
    warped_points = []
    for kp in points:
        warped_points.append(warp_kp_with_bilinear_v2(flow, kp))
    return np.array(warped_points)

def draw_bbox(img, label):
    ### get image size
    img = (img*255).type(torch.uint8)
    w, h = img.shape[2], img.shape[1]
    x_mult = w/100
    y_mult = h/100
    x0, y0, x1, y1 = label["x"]*x_mult, label["y"]*y_mult, (label["x"]+label["width"])*x_mult, (label["y"]+label["height"])*y_mult

    img = draw_bounding_boxes(img, torch.tensor([[x0, y0, x1, y1]]), width=2, colors='red')
    return img.type(torch.float32) / 255

def get_tetrahedron(center, r=0.3):
    points = [
        center,
        (center[0], center[1]+r, center[2]),
        (center[0]+(1/2)**0.5*r, center[1]-0.5*r, center[2]+0.5*r),
        (center[0]-(1/2)**0.5*r, center[1]-0.5*r, center[2]+0.5*r),
        (center[0], center[1]-0.5*r, center[2]-(3/4)**0.5*r),
    ]
    return np.array(points)

def to_3d(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array([[sample_with_binear(depth, kp)] for kp in points])
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_3d_v2(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array([[sample_with_binear_v2(depth, kp)] for kp in points])
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_3d_uvd(points, depth, cmat):
    points = points.reshape(-1, 2)
    depths = np.array(depth)
    # depths = np.array([[depth[int(p[1]), int(p[0])]] for p in points])
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1) * depths
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    cmat = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    points = np.dot(np.linalg.inv(cmat), points.T).T
    points = points[:, :3]
    return points

def to_2d(points, cmat):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = np.dot(cmat, points.T).T
    points = points[:, 0:2] / points[:, 2:3]
    return points

def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])

def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero())[::-1].T.astype(np.float64)
    if len(on) == 0:
        on = np.array((mask==0).nonzero())[::-1].T.astype(np.float64)
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    ### add +-0.5 uniform noises to the samples
    samples = on[sample_ind]
    samples += np.random.uniform(-0.5, 0.5, samples.shape)
    return samples

def get_grasp(samples, depth, cmat, r=5):
    def loss(i):
        return np.linalg.norm(samples - samples[i], axis=1).sum()
    grasp_2d = samples[np.argmin([loss(i) for i in range(len(samples))])]
    neighbor_threshold = r
    neighbors = samples[np.linalg.norm(samples - grasp_2d, axis=1) < neighbor_threshold]
    neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    d = np.median(neighbors_d)
    # print(d)
    # print(grasp_2d)
    return to_3d_uvd(grasp_2d, [d], cmat)

def get_transforms(seg, depth, cmat, flows=[], ransac_tries=100, ransac_threshold=0.5, rgd_tfm_tries=50, rgd_tfm_threshold=1e-3):
    transformss = []
    center_2ds = []
    sampless = []
    samples_2d = sample_from_mask(seg, 500)
    sampless.append(samples_2d)
    samples_3d = to_3d(samples_2d, depth, cmat)
    grasp = get_grasp(samples_2d, depth, cmat)
    # print(grasp.shape)
    # print(samples_3d.shape)
    
    points1_uv = samples_2d
    points1 = samples_3d
    center = grasp
    for i in range(len(flows)):
        flow = flows[i]
        center_uv = to_2d(center, cmat)[0]
        center_2ds.append(center_uv)
        points2_uv = warp_points(flow, points1_uv)
        t0 = time.time()
        _, inliners = ransac(points1_uv, center_uv, points2_uv, ransac_tries, ransac_threshold)
        t1 = time.time()
        # print("inliners:", len(inliners))
        points1_uv = np.array(points1_uv)[inliners]
        points2_uv = np.array(points2_uv)[inliners]
        points1 = np.array(points1)[inliners]
        sampless.append(points2_uv)
        
        solution, mat = solve_3d_rigid_tfm(points1, points2_uv, cmat, rgd_tfm_tries, rgd_tfm_threshold)
        t2 = time.time()

        # print("ransac time:", t1-t0)
        # print("solve time:", t2-t1)
        # print("transform parameters:", solution.x)
        # print("loss:", solution.fun)
        T = get_transformation_matrix(*solution.x)
        
        points1_ext = np.concatenate([points1, np.ones((len(points1), 1))], axis=1)
        points1 = (T @ points1_ext.T).T[:, :3]
        center = (T @ np.concatenate([center, np.ones((1, 1))], axis=1).T).T[:, :3]
        # print("center:", center)
        points1_uv = to_2d(points1, cmat)
        
        transformss.append(solution.x)
    
    return grasp, np.array(transformss), np.array(center_2ds), sampless

def get_inbound_kp_idxs(kps, size):
    h, w = size
    shrink = 4
    return np.array([i for i, kp in enumerate(kps) if kp[0] >= shrink and kp[0] < w-shrink and kp[1] >= shrink and kp[1] < h-shrink])
  
def get_transforms_nav(depth, cmat, flows=[], moving_threshold=1.0, rgd_tfm_tries=30, rgd_tfm_threshold=1e-3):
    transformss = []
    sampless = []
    num_samples = 1000
    seg = (np.linalg.norm(flows[0], axis=2) > moving_threshold)
    # imageio.imsave("seg.png", seg.astype(np.uint8)*255)
    samples_2d = sample_from_mask(seg, num_samples)
    sampless.append(samples_2d)
    samples_3d = to_3d_v2(samples_2d, depth, cmat)

    # print(grasp.shape)
    # print(samples_3d.shape)
    
    points1_uv = samples_2d
    points1 = samples_3d
    for i in range(len(flows)):
        flow = flows[i]
        points2_uv = warp_points_v2(flow, points1_uv)
        inliners = get_inbound_kp_idxs(points2_uv, depth.shape[:2])
        # print("inliners:", len(inliners))
        if len(inliners) < num_samples // 10:
            return np.array(transformss)
        points1_uv = np.array(points1_uv)[inliners]
        points2_uv = np.array(points2_uv)[inliners]
        points1 = np.array(points1)[inliners]
        sampless.append(points2_uv)
        
        solution, mat = solve_3d_rigid_tfm(points1, points2_uv, cmat, rgd_tfm_tries, rgd_tfm_threshold)
        # print(solution.fun)
        t2 = time.time()

        # print("ransac time:", t1-t0)
        # print("solve time:", t2-t1)
        # print("transform parameters:", solution.x)
        # print("loss:", solution.fun)
        T = get_transformation_matrix(*solution.x)
        
        points1_ext = np.concatenate([points1, np.ones((len(points1), 1))], axis=1)
        points1 = (T @ points1_ext.T).T[:, :3]
        # print("center:", center)
        points1_uv = to_2d(points1, cmat)
        inliners = get_inbound_kp_idxs(points1_uv, depth.shape[:2])
        if len(inliners) < num_samples // 10:
            return np.array(transformss)
        points1_uv = np.array(points1_uv)[inliners]
        points1 = np.array(points1)[inliners]
        
        transformss.append(solution.x)
    
    return np.array(transformss)

def transforms2actions(transforms, verbose=False):
    actions = []
    for transform in transforms:
        T = get_transformation_matrix(*transform)
        subgoal = np.matmul(T, np.array([0, 0, 1, 1]))[:3]
        if verbose:
            print("subgoal:", subgoal)

        if np.allclose(subgoal, np.array([0, 0, 1]), atol=1e-3):
            actions.append("Done")
            return actions
        elif subgoal[0] > 0.20:
            actions.append("RotateLeft")
        elif subgoal[0] < -0.20:
            actions.append("RotateRight")
        else:
            actions.append("MoveAhead")
    return actions

def decoding_rules(flow):
    '''
    flow: (T,H,W,2) np.array
    '''
    action_space = ["MoveAhead","RotateLeft","RotateRight","End"]
    
    # # 1. rules in terms of x 
    # points = []
    # for each_flow in flow:
    #     threshold = np.percentile(np.linalg.norm(each_flow, axis=2), 50)
    #     filtered_flow = each_flow[np.linalg.norm(each_flow, axis=2) > threshold]
    #     represent_point = filtered_flow.mean(axis=0)
    #     points.append(represent_point)
    points = reduce(flow, 'b h w c -> b c', 'mean')
    threshold = 90
    action_indices = []
    for i in range(len(points)):
        if points[i][0] > threshold:
            action_indices.append(1)
        elif points[i][0] < -threshold:
            action_indices.append(2)
        else:
            action_indices.append(0)
    
    # 2. rules in terms of angle
    # patch_size = 32
    # max_pooled_flow = max_pool2d(torch.from_numpy(flow).permute(0, 3, 1, 2), kernel_size=patch_size, stride=patch_size).numpy()
    # avg_dx = reduce(max_pooled_flow[:,0,...], 'b h w -> b', 'mean')
    # avg_dy = reduce(max_pooled_flow[:,1,...], 'b h w -> b', 'mean')
    # avg_degree = np.degrees(np.arctan2(avg_dy, avg_dx))
    # action_indices = []
    # for d in avg_degree:
    #     if abs(d) > 135:
    #         action_indices.append(2)
    #     elif abs(d) < 45:
    #         action_indices.append(1)
    #     else:
    #         action_indices.append(0)
    
    # # 3. clustering
    # points = reduce(flow, 'b h w c -> b c', 'mean')
    # # Step 1: Clustering on the x-axis 
    # kmeans = KMeans(n_clusters=3, random_state=42).fit(points[:, :1])
    # # sort ascending label of centers and remap
    # clusters = kmeans.predict(points[:, :1])
    # centers = kmeans.cluster_centers_.flatten()
    # remap = np.argsort(centers)
    # clusters = remap[clusters]
    # centers = centers[remap]

    # # Calculate s and t as midpoints between sorted cluster centers
    # s = (points[clusters == 0, 0].max() + points[clusters == 1, 0].min()) / 2
    # t = (points[clusters == 1, 0].max() + points[clusters == 2, 0].min()) / 2

    # # Label data based on x-value and cluster assignment
    # action_indices = np.where(points[:, 0] < s, 2, np.where(points[:, 0] > t, 1, 0))

    # # Step 2: Identifying outliers on the y-axis using Z-score
    # y_z_scores = zscore(points[:, 1])
    # action_indices[np.abs(y_z_scores) > 3] = 0 # Overwrite labels for outliers, threshold can be adjusted
    
    # scattar plot
    # plt.scatter(avg_direction[:,0], avg_direction[:,1], s=10, c=action_indices, marker='o', alpha=0.5)
    # plt.show()


    actions = [action_space[i] for i in action_indices]
    return actions, action_indices, points

def decode_action_from_flow(flow):
    '''
    flow: (T,H,W,2) np.array
    '''
    actions, action_indices, represent_points = decoding_rules(flow)
    return actions, action_indices, represent_points       

def get_flow_model():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gmflow-pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    parser.add_argument('--feature_channels', type=int, default=128)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--upsample_factor', type=int, default=4)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--ffn_dim_expansion', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--reg_refine', type=bool, default=True)
    parser.add_argument('--task', type=str, default='flow')
    args = parser.parse_args(args=[])
    DEVICE = 'cuda:0'

    model = UniMatch(feature_channels=args.feature_channels,
                        num_scales=args.num_scales,
                        upsample_factor=args.upsample_factor,
                        num_head=args.num_head,
                        ffn_dim_expansion=args.ffn_dim_expansion,
                        num_transformer_layers=args.num_transformer_layers,
                        reg_refine=args.reg_refine,
                        task=args.task).to(DEVICE)
    if not os.path.exists(args.model):
        args.model = os.path.join('scripts/gmflow/gmflow-pretrained', 'gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    checkpoint = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    model.to(DEVICE)
    model.eval()
    model._requires_grad = False
    return model

### predict per frame flow   
def pred_flow_frame(model, frames, stride=1, device='cuda:0', trunk=-1):
    '''
    frames: (T,C,H,W) np.array
    '''
    DEVICE = device 
    model = model.to(DEVICE)
    frames = torch.from_numpy(frames).float()
    images1 = frames[:-1]
    images2 = frames[1:]
    flows = []
    flows_b = []
    for i,(image1, image2) in tqdm(enumerate(zip(images1, images2)),total=len(images1)):
        if i == trunk: break
        image1, image2 = image1.unsqueeze(0).to(DEVICE), image2.unsqueeze(0).to(DEVICE)
        
        attn_splits_list = [2,8]
        inference_size = expected_shape # nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        ori_size = image1.shape[-2:]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # print('before inference: proportionally scale the image to the pre-defined size')
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',align_corners=True)  # (B,C,H,W) -> (B,C,H',W')
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',align_corners=True)  # (B,C,H,W) -> (B,C,H',W')
        
        with torch.no_grad():
            # assert image1.shape[0] % 32 == 0 and image1.shape[1] % 32 == 0, and image1.shape[0] / image1.shape[1] == 3 / 4 is best
            results_dict = model(image1, image2,  # (1, 3, H, W), (1, 3, H, W)
                attn_type='swin',
                attn_splits_list=attn_splits_list,
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task='flow',
                pred_bidir_flow=True,
            )  # (B,C,H',W'), (B,C,H',W') -> (B,2,H',W')
        
        flow_pr = results_dict['flow_preds'][-1]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # print('proportionally scale the flow to the original size')
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',align_corners=True)  # (B,2,H',W') -> (B,2,H,W)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]  
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
                
        flows += [flow_pr[0:1].permute(0, 2, 3, 1).cpu()]  # [(B,H,W,2)]
        flows_b += [flow_pr[1:2].permute(0, 2, 3, 1).cpu()]  # [(B,H,W,2)]
        
    flows = torch.cat(flows, dim=0)
    flows_b = torch.cat(flows_b, dim=0)  # useless
    
    flows = flows.numpy()
    flows_b = flows_b.numpy()
    colors = [flow_to_image(flow) for flow in flows]
    
    return images1, images2, colors, flows, flows_b  # only flow is used, shape [(B,H,W,2)] 

def predict_actions(flow_model, src):
    '''
    src: either a '*.mp4' video file or np.array of (T,H,W,C)
    '''
    if isinstance(src, str):
        src = mp42np(src)
    color_map, flow = pred_flow_frame(flow_model,src.transpose(0,3,1,2))[2:4]
    pred_actions, pred_action_indices, represent_directions = decode_action_from_flow(flow)
    return pred_actions, pred_action_indices

def add_text_on_images(images):
    image_stack = []
    for text, image in enumerate(images):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f'step:{text}', fill='red',font=ImageFont.truetype("arial.ttf", int(0.15 * image.size[1])))
        image_stack.append(np.array(image))
    return np.array(image_stack)

def np2images(nparray, save_dir, preffix=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, frame in enumerate(nparray):
        imageio.imwrite(os.path.join(save_dir,f'{preffix}{i}.jpg'), frame)


if __name__ == '__main__':
    allset = glob(r'/home/zbh/Desktop/zbh/Proj/nav/my/oso/offline-dataset/robothor-dataset/900/train/FloorPlan_Train1_1/rgb_video.mp4')
    
    # 0.config
    expected_shape = [480, 640]
    flow_model = get_flow_model()
    for src_file in allset:
        vis_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/visualization/optical-flow/FloorPlan_Train1_1_flow'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        flow_file = os.path.join(vis_dir,'flow.npy')
        save_file = os.path.join(vis_dir,'flow.mp4')
        
        # 1.save color flow video
        trunk = 100
        nps = mp42np(src_file)
        if os.path.exists(flow_file):
            flow = np.load(flow_file)
            color_map = [flow_to_image(f) for f in flow]
        else:  # generate and save flow
            color_map, flow = pred_flow_frame(flow_model,nps.transpose(0,3,1,2),trunk=trunk)[2:4]
            np.save(flow_file, flow)
        for i_flow in range(trunk):
            if i_flow == 0:
                draw_flow(flow[i_flow], os.path.join(vis_dir,f'flow_{i_flow}.pdf'), tag=' ')
        # np2images(color_map, vis_dir, preffix='flow_')
        # np2images(nps[:trunk], vis_dir)
        pdb.set_trace()


        # 2.load GT action & decode action from flow
        with open(src_file.replace('.mp4','_action.json'), 'r') as f:
            gt_actions = json.load(f)["actions"][:-1]  # remove the last STOP 
        
        pred_actions, pred_action_indices, represent_directions = decode_action_from_flow(flow)
        correct = [gt_actions[i] == pred_actions[i] for i in range(len(gt_actions))]
        # ---Debug---
        # mistake_index = np.where(np.array(correct)==0)[0]
        # print(mistake_index, '\n',represent_points[mistake_index])

        accuracy = np.mean(correct)
        print(f'{os.path.basename(src_file)}\t|\tAccuracy: {sum(correct)} / {len(correct)} = {accuracy}')
        
        with open(src_file.replace('.mp4','_action_pred.json'), 'w') as f:
            json.dump({"length":len(pred_actions), "pred_actions":pred_actions, "pred_action_indices":pred_action_indices, "accuracy":accuracy}, f)

        continue
        # 3.visualize vector field
        
        for i,(origin_image,flow_output,gt_action,pred_action) in enumerate(zip(nps,flow,gt_actions,pred_actions)):
            show(origin_image,save=os.path.join(vis_dir,f'origin_{i}.jpg'),display=False)
            draw_flow(flow_output, os.path.join(vis_dir,f'vis_flow_{i}.jpg'),tag=f'Step:{i} | GT: {gt_action}  Pred: {pred_action}')
        
        # optinal: save vector field matrix
        tensors = []
        for i in range(len(gt_actions)):
            img = cv2.imread(os.path.join(vis_dir,f'vis_flow_{i}.jpg')).transpose(2, 0, 1) / 255
            tensors.append(torch.from_numpy(img).unsqueeze(0))
        tensors = torch.cat(tensors, dim=0).float()
        save_image(tensors, os.path.join(vis_dir,f'vis_flow.jpg'),nrow=4)
        pdb.set_trace()
