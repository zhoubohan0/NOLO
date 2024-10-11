import argparse
import gzip, re
import importlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import random
import sys
import threading
import time
from glob import glob
from typing import Any, Dict
from collections import defaultdict
import ai2thor.controller
import ai2thor.util.metrics
import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai2thor.util.metrics import compute_spl, vector_distance, compute_single_spl
from transformer_policy.policy import load_dataset
from transformer_policy.config import *


# from startx import startx

logger = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.flush = sys.stdout.flush
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logging.getLogger().setLevel(logging.INFO)

ALLOWED_ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]



def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_object_by_type(event_objects, object_type):
    for obj in event_objects:
        if obj['objectId'].split("|")[0] == object_type:
            return obj
    return None

def get_object_by_id(event_objects, object_id):
    for obj in event_objects:
        if obj['objectId'] == object_id:
            return obj
    return None

def np2mp4(nparray, save_file, actions=None, way='ffmpeg', fps=10):
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
    elif way == 'imageio':
        imageio.mimwrite(save_file, nparray, fps=fps, quality=10)  # quality: [0,10]
    else:
        writer = imageio.get_writer(save_file, fps=fps)
        for frame in nparray:
            writer.append_data(frame)
        writer.close()

    if actions is not None:
        np.save(f"{os.path.splitext(save_file)[0]}.npy", actions)
    print(f'successfully collect {save_file}')

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def setup_controller(scene_name):
    # Initialize the controller with the desired configuration
    controller = ai2thor.controller.Controller(
        width=224,
        height=224,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        continuousMode=True,
        snapToGrid=False,
        headless=False,  
        agentMode="locobot",
        save_image_per_frame=True,
    )
    controller.reset(
        scene=scene_name,
        renderObjectImage=True,
        renderClassImage=True,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        renderSemanticSegmentation=False,
        renderFlowImage=False,    
    )
    # controller.step(action="RandomizeMaterials",useTrainMaterials=True, useValMaterials=True)
    return controller

def compute_success_efficiency(episodes_with_golden):
    eval_sum, eval_count = 0.0, 0
    for episode in episodes_with_golden:
        if episode['success']:
            eval_count += 1
            path = episode["path"]
            shortest_path = episode["shortest_path"]
            eval_sum += compute_single_spl(path, shortest_path, episode["success"])
    return eval_sum / eval_count

class VisualGoalNavigation:
    def __init__(self, cfg_file, agent_class, agent_kwargs, render_depth=False):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.config = self.load_config(cfg_file, render_depth)
        if 'nolo' not in self.agent_kwargs["baseline"]:
            self.config["max_steps"] = 200
        # self.setup_env()
        self.controller_kwargs = {
            "commit_id": self.config["thor_build_id"],
            "width": self.config["width"],
            "height": self.config["height"],
            **self.config["initialize"],
        }

        self.current_scene = None
        self.reachable_positions_per_scene = {}

    @staticmethod
    def load_config(cfg_file, render_depth):
        logger.info("Loading configuration from: %s" % cfg_file)
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f.read())
        if render_depth:
            config["initialize"]["renderDepthImage"] = True
        return config

    @staticmethod
    def setup_env():
        if "DISPLAY" not in os.environ:
            xthread = threading.Thread(target=startx)
            xthread.daemon = True
            xthread.start()
            import time

            # XXX change this to use xdpyinfo
            time.sleep(4)

    @staticmethod
    def load_split(dataset_dir):
        episode_list,dataset = [], {}
        for split_path in glob(os.path.join(dataset_dir, "*.json.gz")):
            logger.info("Loading: {path}".format(path=split_path))

            with gzip.GzipFile(split_path, "r") as f:
                episodes = json.loads(f.read().decode("utf-8"))

                # Build a dictionary of the dataset indexed by scene, object_type
                scene_points, points, curr_scene,curr_object = {}, [], None, None
                for data_point in episodes:
                    if curr_object != data_point["object_type"]:
                        scene_points[curr_object] = points
                        curr_object = data_point["object_type"]
                        points = []
                    if curr_scene != data_point["scene"]:
                        dataset[curr_scene] = scene_points
                        curr_scene = data_point["scene"]
                        scene_points = {}
                    
                    points.append(data_point)

                episode_list += episodes
        return episode_list, dataset

    def inference(self, episodes, goal_frame_dir, output_dir, 
                  save_txt=False,save_episodes=False, device='cuda', random_policy=False, cross_mode=False,seed=42, radius=.8):
        seed_everything(int(seed))
        out_queue = []
        if not random_policy: agent = self.agent_class(**self.agent_kwargs).eval()
        # controller = ai2thor.controller.Controller(**self.controller_kwargs)
        scene = episodes[0]["scene"]
        save_dir = os.path.join(output_dir, f"{'cross-' if cross_mode else ''}evaluation{self.agent_kwargs['mode']}_s{seed}")
        os.makedirs(save_dir, exist_ok=True)
        json_file = os.path.join(save_dir, "metrics.json")
        metric_list = read_json(json_file) if os.path.exists(json_file) else []
        if scene in [m["scene_name"] for m in metric_list]:
            print(f"Scene {scene} has been evaluated.")
            return None
        if save_txt: fw = open(os.path.join(save_dir, f"{scene}.txt"), "w")  #time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        
        controller = setup_controller(scene)
        for ie, e in enumerate(episodes):
            # enviorment reset
            # logger.info("Task Start id:{id} scene:{scene} target_object:{object_type} initial_position:{initial_position} rotation:{initial_orientation}".format(**e))
            # print(f'Searching for {e["id"]}......')
            # controller.initialization_parameters["robothorChallengeEpisodeId"] = e["id"]
            # controller.reset(e["scene"],renderObjectImage=True,renderClassImage=True,)
            controller.step(action={
                "action": "TeleportFull",
                **e["initial_position"],
                "rotation": {"x": 0, "y": e["initial_orientation"], "z": 0},
                "horizon": e["initial_horizon"],
                # "standing": True
            })
            # Agent reset
            RGB_collector = [controller.last_event.frame]
            if not random_policy: agent.to(device), agent.reset()#t=int(os.path.splitext(e["goal_image_name"])[0].split("_")[-1]), O0=O0
            episode_metrics = {
                "trajectory" : [{
                    **e["initial_position"],
                    "rotation" : float(e["initial_orientation"]),
                    "horizon" : e["initial_horizon"]
                }],
                "actions_taken" : []
            }

            # target_obj = get_object_by_type(controller.last_event.metadata["objects"], e["object_type"])
            
            target_obj = get_object_by_id(controller.last_event.metadata["objects"], e["object_id"])
            if target_obj is None:
                print(f"Object {e['object_id']} is not found in scene {e['scene']}")
                episode_result = None
            else: # rollout
                total_steps = 0
                stopped = False
                goal_frame = imageio.imread(os.path.join(goal_frame_dir, e["goal_image_name"]))
                while total_steps < self.config["max_steps"] and not stopped:
                    total_steps += 1
                    event = controller.last_event
                    event.metadata.clear()
                    RGB_collector.append(event.frame)
                    # execute action
                    if random_policy: 
                        action = np.random.choice(3) 
                    else:
                        action = agent.act(dict(rgb=event.frame,goal=goal_frame), total_steps)
                    action = ALLOWED_ACTIONS[action]
                    if action not in ALLOWED_ACTIONS:
                        raise ValueError("Invalid action: {action}".format(action=action))
                    # logger.info("Agent action: {action}".format(action=action))

                    # step
                    if len(episode_metrics["actions_taken"]) and not random_policy:
                        last_action = episode_metrics["actions_taken"][-1]
                        if not last_action['success'] and last_action['action'] == action:  # avoid stuck in the same action
                            allow_actions = ALLOWED_ACTIONS[:]
                            allow_actions.remove(action)
                            action = np.random.choice(allow_actions)  # exploration
                    event = controller.step(action=action)
                    episode_metrics["trajectory"].append({
                        **event.metadata["agent"]["position"],
                        "rotation": event.metadata["agent"]["rotation"]["y"],
                        "horizon": event.metadata["agent"]["cameraHorizon"]
                    })
                    episode_metrics["actions_taken"].append({
                        "action": action,
                        "success": event.metadata["lastActionSuccess"]
                    })
                    stopped = vector_distance(event.metadata["agent"]["position"], target_obj["position"]) < radius  #action == "Stop"

                # success detection: 1. target within 1m  2. target visible  3. stopped [optional]
                episode_metrics["dis_to_target"] = vector_distance(event.metadata["agent"]["position"], target_obj["position"])
                episode_metrics["success"] = target_obj["visible"] or episode_metrics["dis_to_target"] < radius#and 1 and stopped 
                episode_result = {
                    "path": episode_metrics["trajectory"],
                    "shortest_path": e["shortest_path"],
                    "success": episode_metrics["success"],
                    "dis_to_target": episode_metrics["dis_to_target"]
                }
                log_info = f"Episode: {ie:03d} | {e['id']}\t|\ttrajectory length: {len(episode_metrics['actions_taken'])}\t|\tdistance to the goal: {episode_metrics['dis_to_target']:.2f} | {'not' if not episode_metrics['success'] else ''} succeeded.\n"
                if save_txt: fw.write(log_info)
                print(log_info,end='')
                if episode_metrics["success"] and 20<total_steps<300:
                    np2mp4(RGB_collector, os.path.join(save_dir, f"{e['id']}.mp4"), way='imageio', fps=10)

            out_queue.append((e["id"], episode_metrics, episode_result))

        controller.stop()
        if save_txt: fw.close()
        metrics, episode_results = {'scene_name':scene, "episodes" : {}}, []
        '''
        [return] metrics.keys() = ['scene_name', 'episodes', 'ep_len', 'success', 'spl', 'dis_to_target', 'category_success', 'ckpt_file']
                 episode_result is a mid result list of ['path', 'shortest_path', 'success'] to calculate SPL and SR
        '''

        for ep_id, episode_metrics, episode_result in out_queue:
            '''
            episode_metrics.keys() = ['trajectory', 'actions_taken', 'success']
            episode_result.keys() = ['path', 'shortest_path', 'success']
            '''
            metrics["episodes"][ep_id] = episode_metrics
            episode_results.append(episode_result)
            
        metrics["ep_len"] = sum([len(em["trajectory"]) for em in metrics["episodes"].values()]) / len(metrics["episodes"])

        clean_episode_results = [r for r in episode_results if r is not None]
        metrics["success"] = sum([r["success"] for r in clean_episode_results]) / len(clean_episode_results)
        metrics["dis_to_target"] = sum([r["dis_to_target"] for r in clean_episode_results]) / len(clean_episode_results)
        metrics["spl"] = compute_spl(clean_episode_results)
        metrics['success_effiency'] = compute_success_efficiency(clean_episode_results)
        print(f"Total Episodes:\t{len(clean_episode_results)}\nSR:\t{metrics['success']}\nSPL:\t{metrics['spl']}\nSuccess Effiency:\t{metrics['success_effiency']}\nEpisode Length:\t{metrics['ep_len']}\nDistance to Target:\t{metrics['dis_to_target']}\n")
        
        # metrics in each category
        category_sr_dict = defaultdict(list)
        for k,v in metrics["episodes"].items():
            category = k.split("_")[-2]
            category_sr_dict[category].append(v["success"])
        category_sr_dict = {k: np.mean(v) for k,v in category_sr_dict.items()}
        print(category_sr_dict)
        metrics['category_success'] = category_sr_dict
        metrics['ckpt_file'] = self.agent_kwargs['ckpt_file'] if not random_policy else 'random_policy'
        
        # save results
        if not save_episodes: metrics.pop("episodes")
        # append metrics to json file
        metric_list.append(metrics)
        save_json(metric_list, json_file)
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Inference script for RoboThor")
    # required setting
    parser.add_argument("--ckpt_file", default="", help="Filepath to agent checkpoint.",)
    parser.add_argument("--scene_name", "-d", default="FloorPlan_Train1_5",help="Filepath to challenge dataset.",)
    parser.add_argument("--output_dir", "-o", default="",help="Filepath to output results to.",)
    parser.add_argument("--mode",default="Q",help="Mode of agent.",choices=["Q", "A", "AQ","P","AP_2","AP_3", "CA", "CQ"],)
    parser.add_argument("--context_type", default='None',help="context type.",)
    # evaluation default setting    
    parser.add_argument("--cfg", "-c",default="scripts/inference_robothor/challenge_config.yaml",help="Filepath to challenge config.",)
    # policy default setting
    parser.add_argument("--random_policy", "-r",default=0,help="Use random policy for evaluation.",)
    parser.add_argument("--baseline", default='nolo-bert')
    parser.add_argument("--seed", default=42,help="Random seed.",)
    args = parser.parse_args()
    

    dataset_dir = f'offline-dataset/robothor-dataset/900/val/{args.scene_name}'
    args.goal_frame_dir = os.path.join(dataset_dir, 'goal_frames')
    dataset = load_dataset(dataset_dir)     # training video frames and actions

    if not args.random_policy:
        if args.baseline == 'nolo-gpt':
            config_dict = read_json(os.path.join(os.path.dirname(args.ckpt_file), 'config.json'))
            if not args.output_dir:
                context_type = '-nocontext' if args.context_type == 'None' else ''
                suffix = re.search(r'policy_(.*?).pth', args.ckpt_file).group(1)
                args.output_dir = os.path.join(os.path.dirname(args.ckpt_file), f'{suffix}{context_type}')
            from transformer_policy.policy import Pi as Pi_nolo_gpt
            agent_kwargs = dict(config=GPTConfig(**config_dict), dataset=dataset, ckpt_file=args.ckpt_file, mode=args.mode, context_type=args.context_type, baseline=args.baseline)
            agent_class = Pi_nolo_gpt
        elif args.baseline == 'nolo-bert':
            config_dict = read_json(os.path.join(os.path.dirname(args.ckpt_file), 'config.json'))
            if not args.output_dir:
                context_type = '-nocontext' if args.context_type == 'None' else ''
                suffix = re.search(r'policy_(.*?).pth', args.ckpt_file).group(1)
                args.output_dir = os.path.join(os.path.dirname(args.ckpt_file), f'{suffix}{context_type}')
            from recbert_policy.vnbert import Pi as Pi_nolo_bert
            agent_kwargs = dict(dataset=dataset, ckpt_file=args.ckpt_file, mode=args.mode, context_type=args.context_type, baseline=args.baseline, temporal_net=config_dict['temporal_net'])
            agent_class = Pi_nolo_bert
        else:
            if not args.output_dir:
                args.output_dir = f'logs/logbaseline-robothor/{args.baseline}'
                os.makedirs(args.output_dir, exist_ok=True)
            agent_kwargs = dict(config=None, dataset=dataset, mode='', baseline=args.baseline, ckpt_file=args.baseline)
            if 'videollava' in args.baseline:
                from baseline_policy.videollava.video_llava_policy import Pi as Pi_videollava
                agent_class = Pi_videollava
            if 'gpt4v' in args.baseline:
                from baseline_policy.gpt4v.gpt4v_policy import Pi_with_context as Pi_gpt4v_context, Pi as Pi_gpt4v
                agent_class = Pi_gpt4v#Pi_gpt4v_context
    else:
        agent_class = lambda **kwargs: None
        agent_kwargs = dict(baseline='nolo_random', mode='')

    vgn = VisualGoalNavigation(args.cfg, agent_class, agent_kwargs)
    episodes = vgn.load_split(dataset_dir)[0]          # testing points
    
    b = time.time()
    cross_mode = ('mp3d' in args.output_dir and 'robothor' in args.ckpt_file) or ('mp3d' in args.ckpt_file and 'robothor' in args.output_dir)

    # 'episodes', 'ep_len', 'success', 'spl', 'dis_to_target', 'category_success', 'ckpt_file'
    metrics = vgn.inference(episodes, goal_frame_dir=args.goal_frame_dir, output_dir=args.output_dir, random_policy=args.random_policy,cross_mode=cross_mode,seed=args.seed)
    print(f'Total time: {time.time()-b:.2f}s')

if __name__ == "__main__":
    main()
