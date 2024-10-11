import os, sys, time, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import json
import random
import multiprocessing as mp
import cv2
import imageio
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
# from omegaconf.dictconfig import DictConfig
from PIL import Image
from scipy.special import softmax
from glob import glob
import torch
import pandas as pd 
from torch.nn.functional import softmax
import quaternion as qt
import habitat
import habitat_sim
from habitat.config.default_structured_configs import FogOfWarConfig,TopDownMapMeasurementConfig
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video,observations_to_image,overlay_frame
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import d3_40_colors_rgb, quat_from_angle_axis, quat_from_two_vectors
from habitat_sim.agent.agent import Agent, AgentConfiguration, AgentState
from collections import defaultdict
from utils.basic_utils import *
from scripts.collect_habitat_all import get_sim_and_action_config, get_pr

from transformer_policy.policy import load_dataset#,Pi as Pi_nolo_gpt


def path_distance(path):
    return sum([np.linalg.norm(np.array(path[i])-np.array(path[i+1])) for i in range(len(path) - 1)])

def compute_single_spl(path, shortest_path, success):
    if not success:
        return 0.0
    li = path_distance(shortest_path)
    pi = path_distance(path)
    spl = 1.0 * li / max(pi, li) if max(pi, li) > 0 else 1.0
    return spl

def initialize_agent_with_pose(sim, position=None, rotation=None, agent_id=0):
    agent_state = AgentState()
    if position is not None:
        agent_state.position = position
    if rotation is not None:
        agent_state.rotation = rotation
    agent = sim.initialize_agent(agent_id, agent_state)
    return agent

def rotvec_to_quat(rotvec) -> qt.quaternion:
    half_theta = np.linalg.norm(rotvec) / 2
    return qt.quaternion(np.cos(half_theta), *(np.sin(half_theta) * rotvec / np.linalg.norm(rotvec)))

def single_process_inference(args, save_txt=False):
    if not args.random_policy:
        dataset = load_dataset(args.data_path)
        if args.baseline == 'nolo-bert':
            config_dict = read_json(os.path.join(os.path.dirname(args.ckpt_file), 'config.json'))
            if not args.output_dir:
                context_type = '-nocontext' if args.context_type == 'None' else ''
                suffix = re.search(r'policy_(.*?).pth', args.ckpt_file).group(1)
                args.output_dir = os.path.join(os.path.dirname(args.ckpt_file), f'{suffix}{context_type}')
                # args.output_dir = os.path.join('logs/logsingle-mp3d-bert/bcq_rank_0.5_9_SA', f'{suffix}{context_type}')
            from recbert_policy.vnbert import Pi as Pi_nolo_bert
            agent_kwargs = dict(dataset=dataset, ckpt_file=args.ckpt_file, mode=args.mode, context_type=args.context_type, baseline=args.baseline, temporal_net=config_dict['temporal_net'])
            agent_class = Pi_nolo_bert
        else:
            if not args.output_dir:
                args.output_dir = f'logs/logbaseline-mp3d/{args.baseline}'
                os.makedirs(args.output_dir, exist_ok=True)
            agent_kwargs = dict(config=None, dataset=dataset, mode='', baseline=args.baseline, ckpt_file=args.baseline)
            if 'videollava' in args.baseline:
                from baseline_policy.videollava.video_llava_policy import Pi as Pi_videollava
                agent_class = Pi_videollava
            if 'gpt4v' in args.baseline:
                from baseline_policy.gpt4v.gpt4v_policy import Pi_with_context as Pi_gpt4v_context
                agent_class = Pi_gpt4v_context
        cross_mode = ('mp3d' in args.output_dir and 'robothor' in args.ckpt_file) or ('mp3d' in args.ckpt_file and 'robothor' in args.output_dir)
        # load policy
        policy = agent_class(**agent_kwargs)
    else:
        cross_mode = False

    # config
    seed_everything(args.seed)
    action_names = ['move_forward', 'turn_left', 'turn_right']#, 'stop'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset_file = glob(os.path.join(args.data_path, "*datasets.json"))[0]
    with open(dataset_file, 'r') as f:
        episodes = json.load(f)
    out_episodes = []

    

    scene = episodes[0]["scene"]
    save_dir = os.path.join(args.output_dir, f"{'cross-' if cross_mode else ''}evaluation{args.mode}_s{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    json_file = os.path.join(save_dir, "metrics.json")
    metric_list = read_json(json_file) if os.path.exists(json_file) else []
    if scene in [m["scene_name"] for m in metric_list]:
        print(f"Scene {scene} has been evaluated.")
        return None
    if save_txt: fw = open(os.path.join(save_dir, f"{scene}.txt"), "w") 

    # load sim
    sim, cfg = get_sim_and_action_config(scene)  # TODO: make it faster and correct  position=[p0[0],1.0,p0[2]],orientation=o0
    category_sr_dict = defaultdict(list)
    stop_radius = 1.  # 1.2
    for ie, e in enumerate(episodes):# interact with environment once
        p0,o0,pg, g = e['initial_position'],e['initial_orientation'],e['target_position'],imageio.imread(os.path.join(args.goal_frame_dir, e["goal_image_name"]))
        objtype = e["object_type"]
        if not args.random_policy: policy.to(device),policy.reset() 
        agent = initialize_agent_with_pose(sim, position=np.array(p0), rotation=rotvec_to_quat(np.array(o0)))
        observations = {**sim.reset(),'collided':False}
        cur_pos, cur_rot = get_pr(agent)

        agent_episode = dict(agent_path=[cur_pos],agent_actions=[],agent_action_indices=[],last_action_success=[])
        t, termination = 0, False
        max_step = 500 if 'nolo' in args.baseline else 200
        while t < max_step and not termination:
            t += 1
            if args.random_policy: 
                action_index = np.random.choice(3) 
            else:
                action_index = policy.act(dict(rgb=observations['color_sensor'][...,:-1],goal=g),t)
            action = action_names[action_index]
            if len(agent_episode["agent_actions"]) and not args.random_policy:
                last_action = agent_episode["agent_actions"][-1]
                ast_action_success = agent_episode["last_action_success"][-1]
                if not ast_action_success and last_action == action:  # avoid stuck in the same action
                    allow_actions = action_names[:]
                    allow_actions.remove(action)
                    action = np.random.choice(allow_actions)  # exploration
            observations = sim.step(action)
            cur_pos, cur_rot = get_pr(agent)
            agent_episode['agent_path'].append(cur_pos)
            agent_episode['agent_actions'].append(action)
            agent_episode['agent_action_indices'].append(action_index)
            agent_episode['last_action_success'].append(not observations['collided'])
            euclidian_distance = np.linalg.norm(np.array(cur_pos)-np.array(pg))
            termination = euclidian_distance < stop_radius

        success = float(termination and t < max_step)
        category_sr_dict[objtype].append(success)

        spl = compute_single_spl(agent_episode['agent_path'], e['shortest_path'], success)
        log_info = f"Episode: {ie:03d} | {e['id']}\t|\tTrajectory Length: {len(agent_episode['agent_actions'])}\t|\tNavigation Error: {euclidian_distance:.2f} | {'not' if not success else ''} succeeded.\n"
        if save_txt: fw.write(log_info)
        print(log_info,end='')
        out_episodes.append(dict(
            id=e['id'],
            success=success,
            spl=spl,
            ep_len=t,
            ne=euclidian_distance,
            initial_position=p0,
            initial_orientation=o0,
            target_position=pg,
            goal_image_name=e['goal_image_name'],
            shortest_path_length=e['shortest_path_length'],
            shortest_path=e['shortest_path'],
        # **agent_episode
        ))


    # save all metrics
    avg_sr = np.mean([e['success'] for e in out_episodes])
    avg_spl = np.mean([e['spl'] for e in out_episodes])
    avg_se = np.mean([e['spl'] for e in out_episodes if e['success']])
    avg_ep_len = np.mean([e['ep_len'] for e in out_episodes])
    avg_ne = np.mean([e['ne'] for e in out_episodes])
    category_sr_dict = {k: np.mean(v) for k,v in category_sr_dict.items()}# metrics in each category

    metrics = {
        'scene_name':scene,
        'ep_len':avg_ep_len,
        'dis_to_target':avg_ne,
        'success':avg_sr, 
        'spl':avg_spl, 
        'success_effiency':avg_se,
        'category_success':category_sr_dict, 
        'ckpt_file':args.ckpt_file if not args.random_policy else 'random_policy'
    }
    print(f"Total Episodes:\t{len(out_episodes)}\nSR:\t{metrics['success']}\nSPL:\t{metrics['spl']}\nSuccess Effiency:\t{metrics['success_effiency']}\nTrajectory Length:\t{metrics['ep_len']}\nNavigation Error:\t{metrics['dis_to_target']}\n")
    metric_list.append(metrics)
    save_json(metric_list, json_file)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Habitat MP3D.")
    # required setting
    parser.add_argument("--ckpt_file", default="", help="Filepath to agent checkpoint.",)
    parser.add_argument("--scene_name", "-d", default="17DRP5sb8fy",help="Filepath to challenge dataset.",)
    parser.add_argument("--output_dir", "-o", default="",help="Filepath to output results to.",)
    parser.add_argument("--mode",default="Q",help="Mode of agent.",choices=["Q", "A", ""],)
    parser.add_argument("--context_type", default='None',help="context type.",)
    # policy default setting
    parser.add_argument("--random_policy", "-r",default=0,help="Use random policy for evaluation.",)
    parser.add_argument("--baseline", default='nolo-bert')
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    b = time.time()
    args.data_path = f'./offline-dataset/mp3d-dataset/900/val/{args.scene_name}'
    args.goal_frame_dir = os.path.join(args.data_path, 'goal_frames')

    single_process_inference(args)
    print(f'Total time: {time.time()-b:.2f}s')