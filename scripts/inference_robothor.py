import argparse
import gzip
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
import yaml
from ai2thor.util.metrics import compute_spl, vector_distance
sys.path.append('/home/zbh/Desktop/zbh/Proj/nav/my/oso')
from rnn_policy.policy import Pi



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

class VisualGoalNavigation:
    def __init__(self, cfg_file, agent_class, agent_kwargs, render_depth=False):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.config = self.load_config(cfg_file, render_depth)

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

    @staticmethod
    def inference_worker(
        worker_ind: int,
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        agent_class: Any,
        agent_kwargs: Dict[str, Any],
        controller_kwargs: Dict[str, Any],
        max_steps: int,
        timeout: int,
        goal_frame_dir,
        output_dir,
    ):
        agent = agent_class(**agent_kwargs)
        controller = ai2thor.controller.Controller(**controller_kwargs)
        # pprint(controller_kwargs)

        fw = open(os.path.join(output_dir, f"eval_{agent_kwargs['mode']}.txt"), "w")  # only 1 worker can record all logs!!
        while True:
            try:
                e = in_queue.get(timeout=timeout)
            except queue.Empty:
                break

            # enviorment reset
            # logger.info("Task Start id:{id} scene:{scene} target_object:{object_type} initial_position:{initial_position} rotation:{initial_orientation}".format(**e))
            # print(f'Searching for {e["id"]}......')
            controller.initialization_parameters["robothorChallengeEpisodeId"] = e["id"]
            controller.reset(e["scene"])
            controller.step(action={
                "action": "TeleportFull",
                **e["initial_position"],
                "rotation": {"x": 0, "y": e["initial_orientation"], "z": 0},
                "horizon": e["initial_horizon"],
                "standing": True
            })
            # Agent reset
            total_steps = 0
            agent.reset()
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
                stopped = False
                rgb_collector = [controller.last_event.frame]
                goal_frame = cv2.imread(os.path.join(goal_frame_dir, e["goal_image_name"]))
                while total_steps < max_steps and stopped is False:
                    total_steps += 1
                    event = controller.last_event
                    event.metadata.clear()
                    # execute action
                    action = agent.act(dict(rgb=torch.from_numpy(event.frame).unsqueeze(0).byte(),goal=torch.from_numpy(goal_frame).unsqueeze(0).byte()))
                    action = ALLOWED_ACTIONS[action]
                    if action not in ALLOWED_ACTIONS:
                        raise ValueError("Invalid action: {action}".format(action=action))
                    # logger.info("Agent action: {action}".format(action=action))

                    # step
                    if len(episode_metrics["actions_taken"]):
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
                    stopped = vector_distance(event.metadata["agent"]["position"], target_obj["position"]) < 1  #action == "Stop"
                    rgb_collector.append(event.frame)

                # success detection: 1. target within 1m  2. target visible  3. stopped [optional]
                episode_metrics["dis_to_target"] = vector_distance(event.metadata["agent"]["position"], target_obj["position"])
                episode_metrics["success"] = target_obj["visible"] or episode_metrics["dis_to_target"] < 1#and 1 and stopped 
                episode_result = {
                    "path": episode_metrics["trajectory"],
                    "shortest_path": e["shortest_path"],
                    "success": episode_metrics["success"],
                    "dis_to_target": episode_metrics["dis_to_target"]
                }
                log_info = f"{e['id']}\t\t|\ttrajectory length: {len(episode_metrics['actions_taken'])}\t|\tdistance to the goal: {episode_metrics['dis_to_target']:.2f} | {'not' if not episode_metrics['success'] else ''} succeeded.\n"
                fw.write(log_info)
                print(log_info,end='')
                if episode_metrics["success"] and total_steps > 2:
                    np2mp4(np.stack(rgb_collector), os.path.join(output_dir,'recording', f'{e["object_type"]}_{total_steps}step.mp4'))

            out_queue.put((e["id"], episode_metrics, episode_result))

        controller.stop()
        fw.close()
        print(f"Worker {worker_ind} Finished.")

    def multi_threads_inference(self, episodes, goal_frame_dir, output_dir, nprocesses=1, test=False, timeout=50):
        send_queue = mp.Queue()
        receive_queue = mp.Queue()

        expected_count = len(episodes)
        np.random.shuffle(episodes[::-1])
        for e in episodes:
            send_queue.put(e)

        processes = []
        for worker_ind in range(nprocesses):
            p = mp.Process(
                target=self.inference_worker,
                kwargs=dict(
                    worker_ind=worker_ind,
                    in_queue=send_queue,
                    out_queue=receive_queue,
                    agent_class=self.agent_class,
                    agent_kwargs=self.agent_kwargs,
                    controller_kwargs=self.controller_kwargs,
                    max_steps=self.config["max_steps"],
                    timeout=timeout,
                    goal_frame_dir=goal_frame_dir,
                    output_dir=output_dir,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.2)

        metrics, episode_results = {"episodes" : {}}, []
        '''
        [return] metrics.keys() = ['episodes', 'ep_len', 'success', 'spl']
                 episode_result is a mid result list of ['path', 'shortest_path', 'success'] to calculate SPL and SR
        '''

        while len(metrics["episodes"]) < expected_count:
            try:
                ep_id, episode_metrics, episode_result = receive_queue.get(timeout=30)
                '''
                episode_metrics.keys() = ['trajectory', 'actions_taken', 'success']
                episode_result.keys() = ['path', 'shortest_path', 'success']
                '''
                metrics["episodes"][ep_id] = episode_metrics
                if not test:
                    episode_results.append(episode_result)
            except TimeoutError:
                print("Went 10 seconds without a new episode result.")
                if all(not p.is_alive() for p in processes):
                    try:
                        ep_id, episode_metrics, episode_result = receive_queue.get(timeout=1)
                        metrics["episodes"][ep_id] = episode_metrics
                        if not test:
                            episode_results.append(episode_result)
                    except TimeoutError:
                        raise RuntimeError("All processes dead but nothing in queue!")

        for p in processes:
            p.join(timeout=1)

        metrics["ep_len"] = sum([len(em["trajectory"]) for em in metrics["episodes"].values()]) / len(metrics["episodes"])

        if not test:
            clean_episode_results = [r for r in episode_results if r is not None]
            metrics["success"] = sum([r["success"] for r in clean_episode_results]) / len(clean_episode_results)
            metrics["dis_to_target"] = sum([r["dis_to_target"] for r in clean_episode_results]) / len(clean_episode_results)
            metrics["spl"] = compute_spl(clean_episode_results)
            logger.info("Total Episodes: {episode_count} Success:{success} SPL:{spl} Episode Length:{ep_len}".format(episode_count=len(clean_episode_results), success=metrics["success"], spl=metrics["spl"], ep_len=metrics["ep_len"]))
        else:
            logger.info("Total Episodes: {episode_count} Episode Length:{ep_len}".format(episode_count=len(episodes), ep_len=metrics["ep_len"]))
        # save results
        save_json(metrics, os.path.join(output_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}_metrics.json'))
        return metrics

    def inference(self, episodes, goal_frame_dir, output_dir, device='cuda', random_policy=False):
        seed_everything(42)
        out_queue = []
        agent = self.agent_class(**self.agent_kwargs)
        controller = ai2thor.controller.Controller(**self.controller_kwargs)
        scene = episodes[0]["scene"]
        fname = os.path.join(output_dir, f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}_{scene}_{self.agent_kwargs['mode']}")
        fw = open(fname+f".txt", "w")  # only 1 worker can record all logs!!
        for ie, e in enumerate(episodes):
            # enviorment reset
            # logger.info("Task Start id:{id} scene:{scene} target_object:{object_type} initial_position:{initial_position} rotation:{initial_orientation}".format(**e))
            # print(f'Searching for {e["id"]}......')
            controller.initialization_parameters["robothorChallengeEpisodeId"] = e["id"]
            controller.reset(e["scene"])
            controller.step(action={
                "action": "TeleportFull",
                **e["initial_position"],
                "rotation": {"x": 0, "y": e["initial_orientation"], "z": 0},
                "horizon": e["initial_horizon"],
                "standing": True
            })
            # Agent reset
            O0 = controller.last_event.frame
            total_steps = 0
            agent.to(device), agent.reset(device=device, t=int(os.path.splitext(e["goal_image_name"])[0].split("_")[-1]), O0=O0)
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
                stopped = False
                rgb_collector = [O0]
                goal_frame = cv2.imread(os.path.join(goal_frame_dir, e["goal_image_name"]))
                while total_steps < self.config["max_steps"] and stopped is False:
                    total_steps += 1
                    event = controller.last_event
                    event.metadata.clear()
                    # execute action
                    act = agent.act3 if agent.n == 3 else agent.act9
                    action = np.random.choice(3) if random_policy else act(dict(rgb=event.frame,goal=goal_frame))
                    action = ALLOWED_ACTIONS[action]
                    if action not in ALLOWED_ACTIONS:
                        raise ValueError("Invalid action: {action}".format(action=action))
                    # logger.info("Agent action: {action}".format(action=action))

                    # step
                    if len(episode_metrics["actions_taken"]):
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
                    stopped = vector_distance(event.metadata["agent"]["position"], target_obj["position"]) < 1  #action == "Stop"
                    rgb_collector.append(event.frame)

                # success detection: 1. target within 1m  2. target visible  3. stopped [optional]
                episode_metrics["dis_to_target"] = vector_distance(event.metadata["agent"]["position"], target_obj["position"])
                episode_metrics["success"] = target_obj["visible"] or episode_metrics["dis_to_target"] < 1#and 1 and stopped 
                episode_result = {
                    "path": episode_metrics["trajectory"],
                    "shortest_path": e["shortest_path"],
                    "success": episode_metrics["success"],
                    "dis_to_target": episode_metrics["dis_to_target"]
                }
                log_info = f"Episode: {ie:03d} | {e['id']}\t|\ttrajectory length: {len(episode_metrics['actions_taken'])}\t|\tdistance to the goal: {episode_metrics['dis_to_target']:.2f} | {'not' if not episode_metrics['success'] else ''} succeeded.\n"
                fw.write(log_info)
                print(log_info,end='')
                if False:#episode_metrics["success"] and 10 <= total_steps <= 50:
                    np2mp4(np.stack(rgb_collector), os.path.join(output_dir,'recording', f'{e["object_type"]}_{total_steps}step.mp4'))

            out_queue.append((e["id"], episode_metrics, episode_result))

        controller.stop()
        fw.close()
        metrics, episode_results = {"episodes" : {}}, []
        '''
        [return] metrics.keys() = ['episodes', 'ep_len', 'success', 'spl']
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
        print("Total Episodes: {episode_count} Success:{success} SPL:{spl} Episode Length:{ep_len}".format(episode_count=len(clean_episode_results), success=metrics["success"], spl=metrics["spl"], ep_len=metrics["ep_len"]))
        
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
        save_json(metrics, fname+f"_metrics.json")
        return metrics

    #utils
    def _change_scene(self, scene):
        if self.current_scene != scene:
            self.current_scene = scene
            self.controller.reset(scene)
            logger.info("Changed to scene: '{scene}'".format(scene=scene))

    def move_to_point(self, datapoint):
        self._change_scene(datapoint["scene"])
        logger.info("Moving to position: {p}, y-rotation: {rot}, horizon: {hor}".format(
            p=datapoint["initial_position"],
            rot=datapoint["initial_orientation"],
            hor=datapoint["initial_horizon"]
        ))
        return self.controller.step(
            action="TeleportFull",
            x=datapoint["initial_position"]["x"],
            y=datapoint["initial_position"]["y"],
            z=datapoint["initial_position"]["z"],
            rotation={"x" : 0, "y" : datapoint["initial_orientation"], "z" : 0},
            horizon=datapoint["initial_horizon"],
            standing=True
        )

    def move_to_random_dataset_point(self, dataset, scene, object_type):
        if scene in dataset:
            if object_type in dataset[scene]:
                datapoint = random.choice(dataset[scene][object_type])
                return self.move_to_point(datapoint)
            else:
                logger.warning(f"No object of type: '{object_type}' for scene: '{scene}', in dataset")
                return None
        else:
            logger.warning("No scene: '{scene}' in dataset".format(scene=scene))
            return None

    def move_to_random_point(self, scene, y_rotation=0, horizon=0):
        if "test" in scene:
            raise RuntimeError("Moving to random points is not posible in test scenes")
        reachable_positions = self._get_reachable_positions_in_scene(scene)
        p = random.choice(reachable_positions)
        return self.move_to_point({
            "initial_position": p,
            "initial_orientation": y_rotation,
            "initial_horizon": horizon,
            "scene" : scene
        })

    def _get_reachable_positions_in_scene(self, scene):
        self._change_scene(scene)
        if scene not in self.reachable_positions_per_scene:
            event_reachable = self.controller.step({
                "action" : "GetReachablePositions",
                "gridSize" : self.config["initialize"]["gridSize"]
            })
            self.reachable_positions_per_scene[scene] = event_reachable.metadata["actionReturn"]
        return self.reachable_positions_per_scene[scene]


def main():
    parser = argparse.ArgumentParser(description="Inference script for RoboThor ObjectNav challenge.")
    # required setting
    parser.add_argument("--ckpt_file", default="", help="Filepath to agent checkpoint.",)
    parser.add_argument("--scene_name", "-d", default="FloorPlan_Train1_1",help="Filepath to challenge dataset.",)
    parser.add_argument("--output_dir", "-o", default="",help="Filepath to output results to.",)
    parser.add_argument("--mode",default="A",help="Mode of agent.",choices=["Q", "A", "AQ","P","AP_2","AP_3", "CA", "CQ"],)
    # evaluation default setting    
    parser.add_argument("--cfg", "-c",default="scripts/inference_robothor/challenge_config.yaml",help="Filepath to challenge config.",)
    parser.add_argument("--agent", "-a",default="nav_agent",help="Relative module for agent definition.",)
    parser.add_argument("--nprocesses", "-n",default=1,type=int,help="Number of parallel processes used to compute inference.",)
    # policy default setting
    parser.add_argument("--backbone", "-b",default="fast_resnet50",help="Visual backbone.",)
    parser.add_argument("--goal_backbone", "-g",default="clip",help="Goal visual backbone.",)
    parser.add_argument("--rnn_type",default="gru")
    parser.add_argument("--enable_feature_matching", "-e",action="store_true",help="Enable feature matching.",)
    parser.add_argument("--entropy_threshold",default=0.5,help="Entropy threshold for AP mode. Above for prototype decision",)
    parser.add_argument("--random_policy", "-r",default=0,help="Use random policy for evaluation.",)
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.ckpt_file)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    agent_kwargs = dict(ckpt_file=args.ckpt_file,mode=args.mode,entropy_threshold=args.entropy_threshold,backbone=args.backbone,goal_backbone=args.goal_backbone,obs_transform=None,rnn_type=args.rnn_type,enable_feature_matching=args.enable_feature_matching,domain='robothor')  # args.scene_name
    dataset_dir = f'offline-dataset/robothor-dataset/900/eval/{args.scene_name}'
    agent_kwargs.update(dict(dataset_dir=dataset_dir,hidden_size=1024,num_recurrent_layers=8))
    vgn = VisualGoalNavigation(args.cfg, Pi, agent_kwargs)
    episodes, dataset = vgn.load_split(dataset_dir)
    args.goal_frame_dir = os.path.join(dataset_dir, 'goal_frames')
    b = time.time()
    metrics = vgn.inference(episodes, goal_frame_dir=args.goal_frame_dir, output_dir=args.output_dir, random_policy=args.random_policy)
    print(f'Total time: {time.time()-b:.2f}s')

if __name__ == "__main__":
    main()
