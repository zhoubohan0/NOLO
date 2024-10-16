import argparse
import gzip
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import os.path as osp
import pdb
import random
from glob import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np
from gmflow.flow2action import Flow2Action, draw_flow, reduce, show
from PIL import Image
from tqdm import tqdm

from utils.basic_utils import *

action_space =  ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]

def append_psuedo_action_labeling(flowActionDecoder, action_indices, data_dir, thre_x=90,thre_y=35, visulize=False):
    # check_and_decode_action_from_video(data_dir)
    if os.path.exists(os.path.join(data_dir, 'points.npy')):
        points = np.load(os.path.join(data_dir, 'points.npy'))
        pred_actions, pred_action_indices, points = flowActionDecoder.decode_action_from_vec(points,thre_x, thre_y)
    else:
        video_frames = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
        pred_actions, pred_action_indices, flow = flowActionDecoder.predict_actions_from_one_video(video_frames,thre_x, thre_y)
        points = reduce(flow, 'b h w c -> b c', 'mean')
        np.save(os.path.join(data_dir, 'points.npy'), points)

    # check the accuracy
    assert len(pred_actions) == len(action_indices)
    accuracy = np.mean([a == b for a, b in zip(pred_action_indices, action_indices)])

    if visulize:
        print(f'acc: {accuracy} |threreshold: {thre_x, thre_y} ')

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        # cmap = plt.cm.get_cmap('tab20', 3)
        ax[0].scatter(points[:,0], points[:,1], c=action_indices)
        # ax[0].vlines(thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # ax[0].vlines(-thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # for i, (x, y) in enumerate(points):
        #     ax[0].text(x, y, str(i), fontsize=8, color=cmap(action_indices[i]))
        ax[0].set_title('action indices')
        ax[1].scatter(points[:,0], points[:,1], c=pred_action_indices)
        # ax[1].vlines(thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # ax[1].vlines(-thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # for i, (x, y) in enumerate(points):
        #     ax[1].text(x, y, str(i), fontsize=8, color=cmap(action_indices[i]))
        ax[1].set_title('pred action indices')
        plt.savefig(os.path.join(data_dir, f'action_prediction_{100*accuracy:.2f}.png'))
        plt.show()

    return pred_actions, pred_action_indices, [thre_x, thre_y], accuracy

def check_video_action(data_dir):
    video_frames = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
    all_data = read_json(os.path.join(data_dir, 'data.json'))
    breakpoint()
    true_actions = all_data['true_actions']
    pred_actions = all_data['pred_actions']
    for i, (frame, true_action, pred_action) in enumerate(zip(video_frames, true_actions, pred_actions)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f'true:{action_space[true_action]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f'pred:{action_space[pred_action]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(f'timestep:{i}', frame)
        if cv2.waitKey(1500) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="offline-dataset/maze-dataset/0")
    args = parser.parse_args()
    seed_everything(42)

    # check_video_action(args.data_dir)

    flowActionDecoder = Flow2Action()
    action_indices = np.load(osp.join(args.data_dir,'action.npy'))
    xrange = range(10, 500, 5)
    yrange = range(10, 500, 1)
    samples = [append_psuedo_action_labeling(flowActionDecoder, action_indices, args.data_dir, thre_x=thre_x, thre_y=thre_y, visulize=thre_x==10 and thre_y==10) for thre_x in xrange for thre_y in yrange]
    pred_actions, pred_action_indices, threshold, accuracy = sorted(samples, key=lambda x: x[-1])[-1]
    print(f'Choose threshold: {threshold} | Accuracy: {accuracy}')
    
    # save the decoded action and data
    all_data={}
    all_data['pred_actions'] = pred_action_indices
    all_data['true_actions'] = action_indices
    all_data['threshold'] = threshold
    all_data['accuracy'] = accuracy
    save_json(all_data, os.path.join(args.data_dir, 'data.json'))