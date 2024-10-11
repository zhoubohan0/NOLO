import numpy as np
import matplotlib.pyplot as plt
import json
import os, re
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings("ignore")

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_heatmap(metric_dict, ax, title):
    '''
    draw (reshaped) heatmap of SR / SPL
    '''
    metric_dict = sorted(metric_dict.items(), key=lambda x: tuple(map(int, re.findall(r'\d+', x[0]))))
    metric = np.array([x[1] for x in metric_dict])
    if not len(metric): return ax
    if len(metric) % 3 == 0:
        metric = metric.reshape(3,-1)
    elif len(metric) % 5 == 0:
        metric = metric.reshape(-1,5)
    sns.heatmap(metric, annot=True, fmt=".4f", cmap='Blues', cbar=False, annot_kws={"size": 30}, ax=ax)
    title += f' (total mean={np.nanmean(metric):.4f})'
    if 'Val' in title:
        title += f'\nlayout mean={np.nanmean(metric,axis=1)}'
        ax.set_yticks(np.arange(3)+0.5)
        ax.set_yticklabels([f'layout{i}' for i in range(1,4)], fontsize=10)
        ax.set_xticks(np.arange(metric.shape[-1])+0.5)
        ax.set_xticklabels(range(1, metric.shape[-1]+1), fontsize=10)
    else:
        ax.set_yticks([])
        ax.set_xticks([])
    ax.set_title(title, fontsize=25)
    return ax   

def reformat_metric_dict(json_file):
    defalut_value = np.nan
    keys = ['success', 'spl', 'ep_len', 'dis_to_target']

    metrics = read_json(json_file)
    domain = 'mp3d'
    if 'robothor' in json_file:
        TrainScenes = [f'FloorPlan_Train{i}_5' for i in range(1, 13)]
        ValScenes = [f'FloorPlan_Val{i}_{j}' for i in range(1, 4) for j in range(1, 6)]
    elif 'habitat' in json_file or 'mp3d' in json_file:
        with open(f'offline-dataset/{domain}-dataset/train.txt', 'r') as f:
            TrainScenes = f.read().splitlines()
        TrainScenes = []
        with open(f'offline-dataset/{domain}-dataset/val.txt', 'r') as f:
            ValScenes = f.read().splitlines()

    TrainMetricDicts, ValMetricDicts = [], []
    for k in keys:
        TrainMetric = {m['scene_name']: m[k] for m in metrics if m['scene_name'] in TrainScenes}
        TrainMetric.update({scene: defalut_value for scene in TrainScenes if scene not in TrainMetric})
        TrainMetricDicts.append((k,TrainMetric))

        ValMetric = {m['scene_name']: m[k] for m in metrics if m['scene_name'] in ValScenes}
        ValMetric.update({scene: defalut_value for scene in ValScenes if scene not in ValMetric})
        ValMetricDicts.append((k,ValMetric))

    return TrainMetricDicts, ValMetricDicts

def standard_metric_dict(json_file):
    keys = ['SR', 'SPL', 'NE', 'TL']
    if '2' in json_file:
        domain = 'robothor' if 'habitat2robothor' in json_file else 'habitat'
    else:
        domain = 'robothor' if 'robothor' in json_file else 'habitat'
    if 'robothor' == domain:
        TrainScenes = [f'FloorPlan_Train{i}_5' for i in range(1, 13)]
        ValScenes = [f'FloorPlan_Val{i}_{j}' for i in range(1, 4) for j in range(1, 6)]
    elif 'habitat' == domain:
        with open('offline-dataset/mp3d-dataset/train.txt', 'r') as f:
            TrainScenes = f.read().splitlines()
        with open('offline-dataset/mp3d-dataset/val.txt', 'r') as f:
            ValScenes = f.read().splitlines()

    metrics = read_json(json_file)
    TrainMetricDicts = [(k, {m['scene']: m[k] for m in metrics if m['scene'] in TrainScenes}) for k in keys]
    ValMetricDicts = [(k, {m['scene']: m[k] for m in metrics if m['scene'] in ValScenes}) for k in keys]
    return TrainMetricDicts, ValMetricDicts

def plot_metrics(json_file):
    '''
    Plot SR and SPL from json_file, metric_names=['success', 'spl']
    '''
    TrainMetricDicts, ValMetricDicts = standard_metric_dict(json_file) if 'final' in json_file else reformat_metric_dict(json_file)
    n = len(TrainMetricDicts)

    fig, ax = plt.subplots(n, 2, figsize=(6*2*n, 8*4))
    for i in range(n):
        key, TrainMetric = TrainMetricDicts[i]
        key, ValMetric = ValMetricDicts[i]
        plot_heatmap(TrainMetric, ax=ax[i,0], title=f'Train {key}')
        plot_heatmap(ValMetric, ax=ax[i,1], title=f'Val {key}')

    plt.savefig(json_file.replace('json', 'jpg'))
    # plt.show()
    fig.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', type=str, default='')
    parser.add_argument('--f2', type=str, default='')
    args = parser.parse_args()
    if os.path.exists(args.f1):
        plot_metrics(args.f1)
    if os.path.exists(args.f2):
        plot_metrics(args.f2)