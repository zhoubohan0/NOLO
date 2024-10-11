import argparse
import json
import os
import os.path as osp
import random
import sys
import time
import warnings
from glob import glob
from pprint import pprint
from typing import TYPE_CHECKING, Union, cast

import attr
import habitat
import habitat_sim
import imageio
import magnum as mn
import numpy as np
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.agent.agent import AgentState
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import d3_40_colors_rgb, quat_from_angle_axis
from matplotlib import pyplot as plt
from omegaconf.dictconfig import DictConfig
from PIL import Image
from scipy.special import softmax

# from Detic.detic_video_example import config_detector, get_parser
from scripts.gmflow.flow2action import (
    Flow2Action,
    SuperGlue2Action,
    draw_flow,
    reduce,
    show,
)

warnings.filterwarnings("ignore")
from utils.basic_utils import *

dummy_path = habitat_sim.ShortestPath()

def example_pointnav_draw_target_birdseye_view():
    # Define NavigationEpisode parameters
    goal_radius = 0.5
    goal = NavigationGoal(position=[10, 0.25, 10], radius=goal_radius)
    agent_position = [0, 0.25, 0]
    agent_rotation = -np.pi / 4

    # Create dummy episode for birdseye view visualization
    dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id="dummy_id",
        scene_id="dummy_scene",
        start_position=agent_position,
        start_rotation=agent_rotation,  # type: ignore[arg-type]
    )

    agent_position = np.array(agent_position)
    # Draw birdseye view
    target_image = maps.pointnav_draw_target_birdseye_view(
        agent_position,
        agent_rotation,
        np.asarray(dummy_episode.goals[0].position),
        goal_radius=dummy_episode.goals[0].radius,
        agent_radius_px=25,
    )
    plt.imshow(target_image)
    plt.title("pointnav_target_image.png")
    plt.show()
    
def example_top_down_map_measure():
    # Create habitat config
    config = habitat.get_config(config_path=os.path.join("habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml",))
    # Create dataset
    dataset = habitat.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True,visibility_dist=5.0,fov=90,),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(env=env,)#goal_radius=config.habitat.task.measurements.success.success_distance
        # Create video of agent navigating in the first episode
        num_episodes = 1
        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            # Create video from images and save to disk
            images_to_video(vis_frames, output_path, video_name, fps=6, quality=10)
            vis_frames.clear()
            # Display video
            vut.display_video(os.path.join(output_path, f"{video_name}.mp4"))

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius=0.2):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(sim=cast("HabitatSim", env.sim),goal_radius=goal_radius,return_one_hot=False,)

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(cast(NavigationEpisode, self.env.current_episode).goals[0].position)

    def reset(self) -> None:
        pass

class GreedyNavAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, sim, object_positions):
        super().__init__()
        self.sim = sim  
        self.shortest_path_follower = ShortestPathFollower(sim=cast("HabitatSim", sim),goal_radius=0.5,return_one_hot=False)
        self.object_positions = object_positions

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        action_index = self.shortest_path_follower.get_next_action(self.cur_target_position)
        return action_index
    
    def reset(self) -> None:
        self.update_target_position()

    def update_target_position(self,):
        # choose the farest object as target
        cur_position = self.sim.agents[0].get_state().position
        distance = np.linalg.norm(self.object_positions - cur_position, axis=1)
        choice = np.random.choice(len(self.object_positions),p=softmax(distance-distance.max()))
        self.cur_target_position = self.object_positions[choice]
        print(f"target_position: {self.cur_target_position}")


def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        sim_cfg.override_scene_light_defaults = settings["override_scene_light_defaults"]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]

    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = settings['position']#[0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = settings['orientation']#[settings["sensor_pitch"],0.0,0.0,]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Create a semantic sensor spec
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = settings['position']
    semantic_sensor_spec.orientation = settings['orientation']
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    sensor_specs = [rgb_sensor_spec, semantic_sensor_spec]
    
    # agent
    from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
    @attr.s(auto_attribs=True, slots=True)
    class MoveAndSpinSpec:
        forward_amount: float
        spin_amount: float
    
    @habitat_sim.registry.register_move_fn(name='move_forward_and_spin',body_action=True)
    class MoveForwardAndSpin(habitat_sim.SceneNodeControl):
        def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: MoveAndSpinSpec):
            forward_ax = (
                np.array(scene_node.absolute_transformation().rotation_scaling())
                @ habitat_sim.geo.FRONT
            )
            scene_node.translate_local(forward_ax * actuation_spec.forward_amount)

            # Rotate about the +y (up) axis
            rotation_ax = habitat_sim.geo.UP
            scene_node.rotate_local(mn.Deg(actuation_spec.spin_amount), rotation_ax)
            # Calling normalize is needed after rotating to deal with machine precision errors
            scene_node.rotation = scene_node.rotation.normalized()

    agent_config = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensor_specs)#height=1.5
    # modify the action space
    agent_config.action_space["move_forward"] = habitat_sim.ActionSpec("move_forward_and_spin", MoveAndSpinSpec(0.25, 0.0))
    agent_config.action_space["turn_left"] = habitat_sim.ActionSpec("move_forward_and_spin", MoveAndSpinSpec(0.0, 30.0))
    agent_config.action_space["turn_right"] = habitat_sim.ActionSpec("move_forward_and_spin", MoveAndSpinSpec(0.0, -30.0))
    return habitat_sim.Configuration(sim_cfg, [agent_config])

def get_path_from_start_to_end(sim, start, end):
    dummy_path.requested_start = start
    dummy_path.requested_end = end
    sim.pathfinder.find_path(dummy_path)
    return [p.astype(float).tolist() for p in dummy_path.points], dummy_path.geodesic_distance

def get_objects(sim):
    '''Print semantic annotation information (id, category, bounding box details) for the current scene in a hierarchical fashion'''
    objects = []
    object_set = set(['cushion', 'appliances', 'floor', 'table', 'counter', 'misc', 'shelving', 'mirror', 'picture', 'door', 'blinds', 'shower', 'bed', 'chest_of_drawers', 'furniture', 'wall', 'curtain', 'sink', 'towel', 'sofa', 'clothes', 'toilet', 'chair', 'lighting', 'ceiling', 'stool', 'cabinet', 'tv_monitor'])
    black_object_set = ['', 'void', 'objects', 'ceiling', 'floor', 'wall']
    for level in sim.semantic_scene.levels:
        # print(f"Level id:{level.id}, center:{level.aabb.center} | dims:{level.aabb.sizes}")
        for region in level.regions:
            # print(f"Region id:{region.id}, category:{region.category.name()} | center:{region.aabb.center}, dims:{region.aabb.sizes}")
            for obj in region.objects:
                if obj.category.name() in black_object_set: continue
                objects.append(dict(
                    id=obj.id,  # <level_id>_<region_id>_<object_id>
                    semanticid=obj.semantic_id,
                    categoryid=obj.category.index(), 
                    category=obj.category.name(), 
                    center=list(obj.aabb.center.astype(float)), 
                    size=list(obj.aabb.sizes.astype(float))),
                ) 
                object_set.add(obj.category.name())
    return objects, list(object_set)

def get_objects2(sim):
    '''Print semantic annotation information (id, category, bounding box details) for the current scene in a hierarchical fashion'''
    objects,instance_id_to_label_id = [], {}
    object_set = set()
    black_object_set = set(["", 'ceiling', 'floor', 'wall', 'void', 'beam', 'objects'])
    # meaningful_object_set = object_set - black_object_set
    for obj in sim.semantic_scene.objects:
        if obj.category.name() in black_object_set: continue
        # instance_id_to_label_id[int(obj.id.split("_")[-1])] = obj.category.index()
        objects.append(dict(
            id=obj.id,  # <level_id>_<region_id>_<object_id>
            semanticid=obj.semantic_id,
            categoryid=obj.category.index(), 
            category=obj.category.name(), 
            center=list(obj.aabb.center.astype(float)), 
            size=list(obj.aabb.sizes.astype(float))),
        ) 
        
        object_set.add(obj.category.name())
    return objects, list(object_set)#, instance_id_to_label_id

def get_pr(agent):
    agent_state = agent.get_state()
    cur_pos = agent_state.position # agent_state.sensor_states['color_sensor'].position
    cur_qua = agent_state.rotation
    cur_rot = cur_qua.vec
    return list(cur_pos.astype(float)), list(cur_rot.astype(float))  # TODO:test quaternion

def calculate_delta_orientation(cur_pos, cur_rot, target_pos):
    delta_orientation = np.arctan2(target_pos[0] - cur_pos[0], target_pos[2] - cur_pos[2]) - cur_rot[1]
    if delta_orientation > np.pi:
        delta_orientation -= 2 * np.pi
    elif delta_orientation < -np.pi:
        delta_orientation += 2 * np.pi
    return abs(delta_orientation)

def convert_semantic_image(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    return semantic_img 

def get_sim_and_action_config(scene_name):
    mp3d_settings = {
        "scene": f'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/mp3d-scenes/mp3d/{scene_name}/{scene_name}.glb' , 
        # "scene": f'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/gibson-scenes/gibson/{scene_name}.glb',
        # 'scene':None,
        # 'scene':'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-sim/data/versioned_data/hm3d-0.2/hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb',
        "default_agent": 0,  # Index of the default agent
        "width": 224,  # Spatial resolution of the observations
        "height": 224,
        "scene_dataset": '/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/mp3d-scenes/mp3d/mp3d.scene_dataset_config.json',
        # "scene_dataset": f'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/gibson-scenes/gibson/gibson.scene_dataset_config.json',
        # "scene_dataset":'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/replica_cad/replicaCAD.scene_dataset_config.json',
        # "scene_dataset":'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-sim/data/versioned_data/hm3d-0.2/hm3d/example/hm3d_annotated_example_basis.scene_dataset_config.json',
        "position": [0.0,1.0,0.0],  # y=Height default 1.5, (x,z) coordinate
        "orientation": [0.0,0.0,0.0],  # x up and down, y left and right, z forward and backward
        "seed": 1,
        "enable_physics": True,  # enable dynamics simulation
    }
    # hm3d_settings = {
    #     "scene":glob(f'/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/hm3d/val/*-{scene_name}/{scene_name}.basis.glb')[0], 
    #     "default_agent": 0,  # Index of the default agent
    #     "width": 224,  # Spatial resolution of the observations
    #     "height": 224,
    #     "scene_dataset": '/home/zbh/Desktop/zbh/Proj/nav/nav-env/habitat/habitat-lab/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config',
    #     "position": [0.0,1.0,0.0],  # y=Height default 1.5, (x,z) coordinate
    #     "orientation": [0.0,0.0,0.0],  # x up and down, y left and right, z forward and backward
    #     "seed": 1,
    #     "enable_physics": True,  # enable dynamics simulation
    # }
    cfg = make_simple_cfg(mp3d_settings)
    sim = habitat_sim.Simulator(cfg)
    return sim, cfg

def find_most_diverse_dataset(counts_object_type, counts_object_id, max_step, select_step):
    stat = [(
        len(set([y for x in counts_object_type[start_t:start_t + select_step] for y in x])),
        len(set([y for x in counts_object_id[start_t:start_t + select_step] for y in x])))
    for start_t in range(max_step - select_step)]
    max_tuple = max(stat)
    return stat.index(max_tuple)

def initialize_agent_with_pose(sim, position=None, rotation=None, agent_id=0):
    agent_state = AgentState()
    if position is not None:
        agent_state.position = np.array([position[0],1.0,position[2]])
    if rotation is not None:
        agent_state.rotation = quat_from_angle_axis(rotation[1], np.array([0, 1, 0]))
    agent = sim.initialize_agent(agent_id, agent_state)
    return agent

def rollout(scene_name, max_step=3600, change_turn=30, select_step=900, position=None, rotation=None, min_object_type=5):
    sim, cfg = get_sim_and_action_config(scene_name)
    action_names = list(cfg.agents[0].action_space.keys())# move_forward, turn_left, and turn_right
    # initialize an agent
    agent = initialize_agent_with_pose(sim,position,rotation)
    observations = {**sim.reset(),'collided':False}
    cur_pos, cur_rot = get_pr(agent)
    RGBs = [observations["color_sensor"][...,:-1]]
    # video_writer = vut.get_fast_video_writer(os.path.join(output_path, "rgb_video.mp4"), fps=30)
    # video_writer.append_data(observations["color_sensor"])
    all_data = {
        'position': [cur_pos],
        'rotation': [cur_rot],
        'action': [],
        'action_indices': [],
        'objects': [],
    }
    
    # Get the static objects
    objects, object_list = get_objects(sim)
    object_list = sorted(object_list)
    object_positions = np.array([o['center'] for o in objects])
    # print(f"object_list: {object_list}")

    # initialize the detic
    # detic_args = get_parser().parse_args()
    # detic_args.custom_vocabulary = object_list
    # detector = config_detector(detic_args)

    # begin to collect
    count_turn, cur_turn = 0, 1
    action_source = [0] * max_step * 100
    counts_object_type, counts_object_id = [[] for _ in range(max_step)] , [[] for _ in range(max_step)]
    t = 0
    while t < max_step:
        # if t % 100 == 0: print(f"step {t}")
        action_index = action_source.pop()# if np.random.uniform() > 0.5 else np.random.choice(3)  
        observations = sim.step(action_names[action_index])
        
        if observations['collided']:  # intervene next action
            while action_source.pop(): pass
            action_source.extend([cur_turn]* np.random.choice(range(1,4),p=[0.5,0.3,0.2]))
            count_turn += 1
            if count_turn % change_turn == 0:
                count_turn = 0
                cur_turn = 3 - cur_turn
            # continue

        cur_pos, cur_rot = get_pr(agent)
        all_data['position'].append(cur_pos)
        all_data['rotation'].append(cur_rot)
        all_data['action'].append(action_names[action_index])
        all_data['action_indices'].append(action_index)
        RGBs.append(observations["color_sensor"][...,:-1])
        # video_writer.append_data(observations["color_sensor"])

        # detect occurance
        detected_semantic_ids = np.unique(observations['semantic_sensor'])
        # predictions = detector.predict_image(observations["color_sensor"][...,:-1])
        euclidian_distance = np.linalg.norm(object_positions - np.array(cur_pos), axis=1)
        for i in range(len(objects)): 
            objecti = objects[i]
            # delta_orientation = calculate_delta_orientation(cur_pos, cur_rot, objecti['center'])
            if euclidian_distance[i] < 1 and objecti['semanticid'] in detected_semantic_ids: # near and visible
                #object_i_info['category'] in predictions['object_names'] 
                #delta_orientation < np.pi / 2

                # semantic_categories = [instance_id_to_label_id.get(i,0) for i in np.unique(observations['semantic_sensor'])]
                # print(f'step {t}\t| target object: {objecti["category"], objecti["id"], objecti["semanticid"], objecti["categoryid"]}',{euclidian_distance[i]})
                all_data['objects'].append(dict(
                    timestep=t,
                    position=objecti['center'],
                    distance=euclidian_distance[i],
                    objectId=objecti['id'],
                    objectType=objecti['category'],
                ))
                counts_object_type[t].append(objecti['category'])
                counts_object_id[t].append(objecti['id'])
        t += 1
        
    # choose select_step with the most (num_object_type, num_object_id)
    start_t = find_most_diverse_dataset(counts_object_type, counts_object_id, max_step, select_step)
    # modify all_data to remove the first start_t steps and adjust the timestep in all_data['objects']
    all_data['position'] = all_data['position'][start_t:start_t + select_step]
    all_data['rotation'] = all_data['rotation'][start_t:start_t + select_step]
    all_data['action'] = all_data['action'][start_t:start_t + select_step]
    all_data['action_indices'] = all_data['action_indices'][start_t:start_t + select_step]
    all_data['objects'] = [dict(o, timestep=o['timestep'] - start_t) for o in all_data['objects'] if start_t<=o['timestep'] < start_t + select_step]
    RGBs = RGBs[start_t: start_t + select_step + 1]
    assert len(RGBs) == select_step + 1
    # Save ALL
    num_object_id = len(set([o['objectId'] for o in all_data['objects']]))
    num_object_type = len(set([o['objectType'] for o in all_data['objects']]))
    print(f'Collect from timestep {start_t:04d} to {start_t + select_step:04d}, (num_object_type, num_object_id): ({num_object_type:03d}, {num_object_id:03d})')
    # close sim
    if num_object_type < min_object_type:
        seed_everything(random.randint(0, 1000))
        restart_t = random.randint(1, args.select_step)
        position = sim.pathfinder.get_random_navigable_point()
        rotation = all_data['rotation'][restart_t]
        seed_everything(42)
    sim.close()
    return RGBs, all_data, (num_object_type, num_object_id), position, rotation  # sort by the num_object_type first and then num_object_id

def append_psuedo_action_labeling(flowActionDecoder: Flow2Action, action_indices, data_dir, thre_x=90,thre_y=35, visulize=False):
    # check_and_decode_action_from_video(data_dir)
    if os.path.exists(os.path.join(data_dir, 'points.npy')):
        points = np.load(os.path.join(data_dir, 'points.npy'))
        pred_actions, pred_action_indices, points = flowActionDecoder.decode_action_from_vec(points,thre_x, thre_y)
    else:
        video_frames = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
        pred_actions, pred_action_indices, flow = flowActionDecoder.predict_actions_from_one_video(video_frames,thre_x, thre_y)
        points = flow.mean(1).mean(1)#reduce(flow, 'b h w c -> b c', 'mean')
        np.save(os.path.join(data_dir, 'points.npy'), points)

    # check the accuracy
    assert len(pred_actions) == len(action_indices)
    accuracy = np.mean([a == b for a, b in zip(pred_action_indices, action_indices)])

    if visulize:
        print(f'acc: {accuracy} |threreshold: {thre_x, thre_y} ')

        fig, ax = plt.subplots(1,2,figsize=(12,6))
        # cmap = plt.cm.get_cmap('tab20', 3)
        ax[0].scatter(points[:,0], points[:,1], c=action_indices)
        ax[0].vlines(thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        ax[0].vlines(-thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # for i, (x, y) in enumerate(points):
        #     ax[0].text(x, y, str(i), fontsize=8, color=cmap(action_indices[i]))
        ax[0].set_title('action indices')
        ax[1].scatter(points[:,0], points[:,1], c=pred_action_indices)
        ax[1].vlines(thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        ax[1].vlines(-thre_x, -thre_y, thre_y, colors='r', linestyles='dashed')
        # for i, (x, y) in enumerate(points):
        #     ax[1].text(x, y, str(i), fontsize=8, color=cmap(action_indices[i]))
        ax[1].set_title('pred action indices')
        plt.savefig(os.path.join(data_dir, f'action_prediction_{100*accuracy:.2f}.png'))
        fig.clf()
        # plt.show()
    # print(f'accuracy: {accuracy:.3f} | threshold: {thre_x, thre_y}')
    return pred_actions, pred_action_indices, [thre_x, thre_y], accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_step", type=int, default=3600)
    parser.add_argument("--select_step", type=int, default=900)
    parser.add_argument("--begin_i", type=int, default=0)
    parser.add_argument("--end_i", type=int, default=90)
    parser.add_argument('--superglue', action="store_true")
    args = parser.parse_args()
    
    # prepare the data
    fcts = [10,30,50,70]#,100
    with open('offline-dataset/mp3d-dataset/train.txt', 'r') as f:
        train_name_list = f.read().splitlines()
    with open('offline-dataset/mp3d-dataset/val.txt', 'r') as f:
        val_name_list = f.read().splitlines()
    # dataset_dir = 'offline-dataset/hm3d-dataset'
    # with open(f'{dataset_dir}/all.txt', 'r') as f:
    #     scene_name_list = f.read().splitlines() 
    seed_everything(42)
    
    if args.superglue:
        action_decoder = SuperGlue2Action()
        for scene_name, category in [(s,'train')for s in train_name_list] + [(s,'val')for s in val_name_list]:
            data_dir=f'offline-dataset/mp3d-dataset/900/{category}/{scene_name}'
            all_data = read_json(osp.join(data_dir, 'data.json'))
            if 'superglue_action_indices' in all_data: continue
            video_data = mp42np(osp.join(data_dir, 'rgb_video.mp4'))
            GT_actions = all_data["action_indices"]
            if osp.exists(osp.join(data_dir, 'superglue_points.npy')):
                superglue_vecs = np.load(osp.join(data_dir, 'superglue_points.npy'))
            else:
                superglue_vecs = action_decoder.calculate_vector_from_superglue(video_data)
                np.save(osp.join(data_dir, 'superglue_points.npy'), superglue_vecs)

            # fig = plt.figure()
            # plt.scatter(*superglue_vecs.T,c=GT_actions)
            # plt.show()

            samples = []
            for thre_x in range(10, 100, 1):
                for thre_y in range(10, 100, 1):
                    superglue_action_indices = action_decoder.decode_action_from_vec(superglue_vecs,thre_x, thre_y)
                    superglue_accuracy = np.mean([a == b for a, b in zip(superglue_action_indices, GT_actions)])
                    samples.append((superglue_action_indices, (thre_x, thre_y), superglue_accuracy))
            superglue_action_indices, superglue_threshold, superglue_accuracy = sorted(samples, key=lambda x: x[-1])[-1]
            print(f'Scene: {scene_name} | Choose threshold: {superglue_threshold} | Accuracy: {superglue_accuracy}')

            # augment the data.json
            all_data["superglue_action_indices"] = superglue_action_indices
            all_data["superglue_threshold"] = superglue_threshold
            all_data["superglue_accuracy"] = superglue_accuracy
            save_json(all_data, osp.join(data_dir, 'data.json'))
        sys.exit(0)

    for i in range(args.begin_i, args.end_i):
        scene_name = scene_name_list[i]
        beg = time.time()
        split = 'train'# if scene_name in train_name_list else 'val'
        data_dir=f'{dataset_dir}/{args.select_step}/{split}/{scene_name}'
        if not osp.exists(data_dir):
            os.makedirs(data_dir)
        print(f'{scene_name}({split}) start collecting data...')
        # step 1: collect the most diverse data and record, output the rgb_video.mp4 and data.json
        if not osp.exists(osp.join(data_dir, 'rgb_video.mp4')) or not osp.exists(osp.join(data_dir, 'data.json')):
            # sample and save the most diverse data
            samples, num_object_type, min_object_type, p0, o0 = [], 0, 5, None, None
            for _ in range(3):
                try_max_step = args.max_step + _ * 2000
                for fct in fcts:
                    RGBs, all_data, (num_object_type, num_object_id),p0, o0 = rollout(scene_name, try_max_step, fct, args.select_step,p0,o0,min_object_type)
                    samples.append((RGBs, all_data, (num_object_type, num_object_id)))
                samples.sort(key=lambda x: x[-1], reverse=True)
                samples = samples[:1]
                if num_object_type >= min_object_type: break
            RGBs, all_data, (num_object_type, num_object_id) = samples[0]    
            all_data['num_object_type'] = num_object_type
            all_data['num_object_id'] = num_object_id
            # save the video 
            imageio.mimwrite(osp.join(data_dir, 'rgb_video.mp4'), RGBs, fps=10)
            save_json(all_data, os.path.join(data_dir, 'data.json'))   
            print(f'Collected object types:{num_object_type} | object ids:{num_object_id}')
        
        RGBs = mp42np(osp.join(data_dir, 'rgb_video.mp4'))
        all_data = read_json(osp.join(data_dir, 'data.json'))

        # step 2: tune the threshold for psuedo action labeling, output the points.npy and save threshold, labeled actions to the data.json
        if 'threshold' not in all_data:
            flowActionDecoder = Flow2Action()
            action_indices = np.array(all_data['action_indices'])
            xrange = range(30, 91, 1)
            yrange = range(10, 41, 1)
            pred_actions, pred_action_indices, threshold, accuracy = sorted([
                append_psuedo_action_labeling(flowActionDecoder, action_indices, data_dir, thre_x=thre_x, thre_y=thre_y,visulize=False) for thre_x in xrange for thre_y in yrange],
            key=lambda x: x[-1])[-1]
            print(f'Choose threshold: {threshold} | Accuracy: {accuracy}')

            # save the decoded action and data
            all_data['pred_actions'] = pred_actions
            all_data['pred_action_indices'] = pred_action_indices
            all_data['threshold'] = threshold
            all_data['accuracy'] = accuracy
            save_json(all_data, os.path.join(data_dir, 'data.json'))   


        # step 3: make testing points, output directory: goal_frames, and {scene_name}_datasets.json
        # if split != 'val': continue
        if not osp.exists(osp.join(data_dir, f'{scene_name}_datasets.json')) or not osp.exists(osp.join(data_dir, 'goal_frames')):
            goal_dir = os.path.join(data_dir, 'goal_frames')
            os.makedirs(goal_dir, exist_ok=True)
            positions = all_data['position']
            rotations = all_data['rotation']
            out_episodes = []
            count_dict = {obj['objectType']:0 for obj in all_data['objects']}
            N_EACH, DIS_THRE = 10, 2
            repeat = 0
            sim, cfg = get_sim_and_action_config(scene_name)
            while repeat < 2:
                for o in all_data['objects']:
                    timestep, objtype, objid = o['timestep'], o['objectType'], o['objectId'] 
                    target_position = positions[timestep]
                    goal_image_name = f'{scene_name}_{objtype}_{timestep}.png'
                    extent = 100
                    t = np.random.choice(np.arange(max(timestep-extent,0),min(timestep+extent,len(RGBs)-1)))
                    initial_position = positions[t]
                    initial_rotation = rotations[t]
                    shortest_path, shortest_path_length = get_path_from_start_to_end(sim, initial_position, target_position)
                    # print(f'shortest_path_length: {shortest_path_length:.3f}')
                    if shortest_path and shortest_path_length > DIS_THRE and count_dict[objtype] < N_EACH:
                        out_episodes.append({
                            "id": f'{scene_name}_{objtype}_{count_dict[objtype]}',
                            "scene": scene_name,
                            "object_type": objtype,
                            "object_id": objid,
                            "initial_position": initial_position,
                            "initial_orientation": initial_rotation,
                            "start_timestep": int(t),
                            "goal_image_name":goal_image_name,
                            "target_position": target_position,
                            "euclidean_distance": shortest_path_length,
                            "shortest_path_length": shortest_path_length,
                            "shortest_path": shortest_path,
                        })
                        count_dict[objtype] += 1
                        imageio.imwrite(os.path.join(goal_dir,goal_image_name),RGBs[timestep])   # save goal image
                repeat += all([v == N_EACH or v == 0 for v in count_dict.values()])

            out_episodes = sorted(out_episodes,key=lambda x:x['id'])
            with open(osp.join(data_dir, f'{scene_name}_datasets.json'), 'w') as f:
                json.dump(out_episodes, f, indent=4)
            print(f'{len(out_episodes)} testing episodes have been constructed in {data_dir}!')
            sim.close()
        print(f'{scene_name} done in {time.time()-beg:.2f} s')

