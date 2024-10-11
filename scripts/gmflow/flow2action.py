import json
import os
import pdb
import random
import shutil
from argparse import ArgumentParser
from glob import glob

import cv2
import imageio
import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from PIL import Image
from superglue.superglue_matching import Matching
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm

from .unimatch.flow_viz import flow_to_image
from .unimatch.unimatch import UniMatch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# utils
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
    mask_freq[::4, ::8] = 1# regularily sample mask
    mask_new = np.logical_and(mask_per, mask_freq)


    # 应用筛选，准备可视化数据 
    x_filtered_new = x[mask_new] 
    y_filtered_new = y[mask_new] 
    u_filtered_new = u[mask_new] 
    v_filtered_new = v[mask_new] 
    gradient_filtered_new = rad[mask_new] 
    # 可视化筛选后的向量 
    plt.figure() 
    # make arrow larger and transparent
    quiver_new = plt.quiver(x_filtered_new, y_filtered_new, u_filtered_new, v_filtered_new, gradient_filtered_new, scale=200, scale_units='inches', alpha=0.5)
    plt.colorbar(quiver_new, label='Gradient Magnitude') 
    if not tag:
        tag = f'Visualization of Top {100-per}% Gradient Vectors'
    plt.title(tag) 

    # plt.xlim(0, W)
    # plt.ylim(0, H)
    plt.gca().invert_yaxis() 
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


class Flow2Action:
    def __init__(self) -> None:
        self.flow_model = self.get_flow_model()
        self.expected_shape = [480, 640]
        self.action_space = ["MoveAhead","RotateLeft","RotateRight","Stop"]

    def decode_action_from_flow(self, flow, thre_x, thre_y):
        '''
        flow: (T,H,W,2) np.array
        '''
        points = flow.mean(1).mean(1)#reduce(flow, 'b h w c -> b c', 'mean')
        return self.decode_action_from_vec(points, thre_x, thre_y)

    def decode_action_from_vec(self, points, thre_x=65, thre_y=60):
        '''
        points: (T,2) np.array
        '''
        action_indices = []
        for i in range(len(points)):
            if points[i][0] > thre_x and abs(points[i][1]) < thre_y:# or abs(points[i][1]) > thre_y:
                action_indices.append(1)
            elif points[i][0] < -thre_x and abs(points[i][1]) < thre_y:
                action_indices.append(2)
            else:
                action_indices.append(0)
        actions = [self.action_space[i] for i in action_indices]
        actions, action_indices = self.post_correction(actions, action_indices)
        return actions, action_indices, points

    def post_correction(self, actions, action_indices):
        # plt.scatter(range(len(action_indices)), action_indices)
        # plt.show()
        return actions, action_indices

    @staticmethod
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
            args.model = os.path.join('gmflow', args.model)
            if not os.path.exists(args.model):
                args.model = os.path.join('scripts', args.model)
        checkpoint = torch.load(args.model, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])

        model.to(DEVICE)
        model.eval()
        model._requires_grad = False
        return model


    def pred_flow_frame(self, frames, DEVICE='cuda:0', trunk=-1):
        '''
        frames: (T,C,H,W) np.array
        '''
        self.flow_model = self.flow_model.to(DEVICE)
        frames = torch.from_numpy(frames).float()
        images1 = frames[:-1]
        images2 = frames[1:]
        flows = []
        flows_b = []
        for i,(image1, image2) in tqdm(enumerate(zip(images1, images2)),total=len(images1)):
            if i == trunk: break
            image1, image2 = image1.unsqueeze(0).to(DEVICE), image2.unsqueeze(0).to(DEVICE)
            
            attn_splits_list = [2,8]
            inference_size = self.expected_shape # nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
            ori_size = image1.shape[-2:]
            
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                # print('before inference: proportionally scale the image to the pre-defined size')
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',align_corners=True)  # (B,C,H,W) -> (B,C,H',W')
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',align_corners=True)  # (B,C,H,W) -> (B,C,H',W')
            
            with torch.no_grad():
                # assert image1.shape[0] % 32 == 0 and image1.shape[1] % 32 == 0, and image1.shape[0] / image1.shape[1] == 3 / 4 is best
                results_dict = self.flow_model(image1, image2,  # (1, 3, H, W), (1, 3, H, W)
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


    def predict_actions_from_one_video(self, src, thre_x=65, thre_y=60):
        '''
        src: either a '*.mp4' video file or np.array of (T,H,W,C)
        '''
        if isinstance(src, str):
            src = mp42np(src)
        color_map, flow = self.pred_flow_frame(src.transpose(0,3,1,2))[2:4]

        pred_actions, pred_action_indices, represent_directions = self.decode_action_from_flow(flow, thre_x, thre_y)
        return pred_actions, pred_action_indices, flow
    

    def predict_actions_from_videos(self, src_files, save_dir='', debug=False):
        '''
        src_files: list of '*.mp4' video files
        '''
        for src_file in src_files:
            pred_actions, pred_action_indices, flow = self.predict_actions_from_one_video(src_file)
            if debug:# save flow 
                flow_file = src_file.replace('.mp4','_flow.npy')
                np.save(flow_file, flow)
            # save actions
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                action_file = os.path.join(save_dir, os.path.basename(src_file).replace('.mp4','_action_pred.npy'))
            action_file = src_file.replace('.mp4','_action_pred.json')
            with open(action_file, 'w') as f:
                json.dump({"length":len(pred_actions), "pred_actions":pred_actions, "pred_action_indices":pred_action_indices}, f)
            print(f'successfully save {action_file}')



class SuperGlue2Action:
    def __init__(self) -> None:
        self.matcher = Matching().cuda()
        self.conf_threshold = 0.6
        self.max_matched_pts=1000

    def calculate_vector_from_superglue(self,video_frames,):
        pred_actions = []
        for i in tqdm(range(len(video_frames)-1), total=len(video_frames)-1):
            try:
                pred = self.matcher(dict(
                    image0=rgb_to_grayscale(torch.from_numpy(video_frames[i:i+1]).permute(0,3,1,2)).cuda().float()/255.0,
                    image1=rgb_to_grayscale(torch.from_numpy(video_frames[i+1:i+2]).permute(0,3,1,2)).cuda().float()/255.0,
                ))
                kpts0, kpts1 = pred['keypoints0'][0], pred['keypoints1'][0]
                matches, confidence = pred['matches0'][0], pred['matching_scores0'][0]
                valid = torch.bitwise_and(matches > -1, confidence > self.conf_threshold)
                sorted = torch.argsort(confidence[valid], descending=True)[:self.max_matched_pts]
                mkpts0 = kpts0[valid][sorted]
                mkpts1 = kpts1[matches[valid][sorted].long()]
                delta = mkpts1 - mkpts0
                dominent_vec = delta.mean(0)
            except Exception as e:
                dominent_vec = pred_actions[-1]
            pred_actions.append(dominent_vec)
        return torch.stack(pred_actions, 0).detach().cpu().numpy()
    
    def decode_action_from_vec(self, points, thre_x=65, thre_y=60):
        '''
        points: (T,2) np.array
        '''
        action_indices = []
        for i in range(len(points)):
            if points[i][0] > thre_x and abs(points[i][1]) < thre_y:# or abs(points[i][1]) > thre_y:
                action_indices.append(1)
            elif points[i][0] < -thre_x and abs(points[i][1]) < thre_y:
                action_indices.append(2)
            else:
                action_indices.append(0)
        return action_indices