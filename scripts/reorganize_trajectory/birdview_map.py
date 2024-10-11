import json
import os
import pdb
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def decode_xy_from_action(action_sequence, move_param={'forward': 0.25, 'left': 30, 'right': 30}, start=(0, 0, 0)):
    '''
    start: (x, y, angle)
    '''
    x, y, angle = start
    coordinates2D = [(x, y)]
    for action in action_sequence:
        if action == 0:  # 前进
            x += move_param['forward'] * np.cos(np.radians(angle))
            y += move_param['forward'] * np.sin(np.radians(angle))
            coordinates2D.append((x, y))
        elif action == 1:
            angle += move_param['left']
        elif action == 2:
            angle -= move_param['right']
    return np.array(coordinates2D)

def plot_batch_sequences(sequences, nrow=1, ncol=1, save='',tags=[]):
    """
    绘制一组序列的轨迹，每个序列起点是圆环，终点是五角星，线条加粗，圆点变厚，颜色区分。
    参数:
    - sequences: Lists of coordinat sequences
    """
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 5))  
    if nrow * ncol == 1:
        axs = np.array([[axs]])
    elif nrow == 1 or ncol == 1:
        axs = axs.reshape(nrow, ncol)


    for i in range(len(sequences)):
        ax = axs.flat[i]
        coordinates = sequences[i]
        if isinstance(coordinates, tuple):
            xzr0, action_seq = coordinates
            coordinates = decode_xy_from_action(action_seq, move_param, start=xzr0)

        # visualize each trajectory
        xs, ys = zip(*coordinates)
        ax.quiver(xs[:-1], ys[:-1], np.diff(xs), np.diff(ys), scale_units='xy', angles='xy', scale=1, color='black',
                  width=0.01, headwidth=3, headlength=3, headaxislength=3)
        ax.plot(xs[1:-1], ys[1:-1], '-o', markersize=10, linewidth=2, color='blue', markerfacecolor='red',
                markeredgecolor='black', alpha=0.5)

        # 绘制起点和终点特殊标记
        ax.plot(xs[0], ys[0], 'o', markersize=15, color='green', markeredgewidth=1, markeredgecolor='black')  # 起点
        ax.plot(xs[-1], ys[-1], '*', markersize=15, color='yellow', markeredgewidth=1, markeredgecolor='black')  # 终点

        # 每个点右边标记序号
        for j, (x, y) in enumerate(coordinates):
            ax.text(x + 0.04, y, str(j), fontsize=10, color='black')

        # ax.axis('equal')
        ax.set_xlim(min(xs) - 0.25, max(xs) + 0.25)
        ax.set_ylim(min(ys) - 0.25, max(ys) + 0.25)
        ax.set_title(tags[i], fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()
        plt.cla(), plt.clf(), plt.close()


if __name__ == '__main__':
    move_param = {'forward':0.25, 'left':30, 'right':30}
    pred_action_file = 'offline-dataset/robothor-dataset/900/FloorPlan_Train1_1/data.json'

    with open(pred_action_file, 'r') as f:
        data = json.load(f)
        action_indices = data['action_indices']
        pred_action_indices = data['pred_action_indices']
        xz = [[p['x'],p['z']] for p in data['position']]
        rotation = data['rotation']

    for _ in range(20):
        b = np.random.choice(len(action_indices))
        t = np.random.choice(range(1,10))
        e = min(b + t, len(action_indices))
        seq1 = xz[b:e+1]
        seq2 = (xz[b] + [90-rotation[b]['y']],pred_action_indices[b:e])
        plot_batch_sequences([seq1,seq2], nrow=1, ncol=2,tags=[f'real:{b}->{e}',f'pred:{b}->{e}'])