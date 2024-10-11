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
from ai2thor.controller import Controller
from ai2thor.util.metrics import (
    get_shortest_path_to_object,
    get_shortest_path_to_object_type,
    path_distance,
    vector_distance,
)
from gmflow.flow2action import Flow2Action, SuperGlue2Action, draw_flow, reduce, show
from PIL import Image
from tqdm import tqdm

from utils.basic_utils import *

action_space =  ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]


def decode_action_from_pose(rotation, threshold=10):
    '''
    For debugging
    postures: list of dict
    return: list of str
    '''
    actions, action_indices = [], []
    # Calculate delta posture
    for i,(lhs, rhs) in enumerate(zip(rotation[:-1], rotation[1:])):
        # print(f'step:{i} | ', {k: rhs[k] - lhs[k] for k in rhs.keys()}) # Debug: Calculate the delta action
        delta_ry = rhs['y'] - lhs['y']
        if abs(delta_ry) > 50:  # limit the range of delta_ry to about [-30,30]
            delta_ry = 360 - abs(delta_ry) if delta_ry < 0 else abs(delta_ry) - 360
        if delta_ry > threshold:
            action_indice = 2
        elif delta_ry < -threshold:
            action_indice = 1
        else:
            action_indice = 0
        action_indices.append(action_indice)
    # action_indices += [3]  # Add the last action
    actions = [action_space[i] for i in action_indices]
    return actions, action_indices



def rollout(scene_name,max_step=3600,change_turn=30):
    '''
    input: None
    output: data_dir/rgb_video.mp4, data_dir/data.json, data_dir/points.npy (middle result)
    '''
    rgb_writer = []
    # rgb_writer = imageio.get_writer(osp.join(data_dir, 'rgb_video.mp4'), fps=10)
    # depth_writer = np.empty((0, 480, 640), dtype=np.float32)
    # seg_writer = imageio.get_writer(osp.join(data_dir, 'semantic_video.mp4'))
    # ins_writer = imageio.get_writer(osp.join(data_dir, 'instance_video.mp4'))

    controller = setup_controller(scene_name)

    # Get the reachable positions
    # all_object_id = {o['objectId']:[] for o in controller.last_event.metadata['objects']}
    # reachable_positions = controller.step(action="GetReachablePositions").metadata['reachablePositions']
    # o2c, c2o = controller.last_event.object_id_to_color, controller.last_event.color_to_object_id
    # visited = np.zeros(len(reachable_positions), dtype=bool)
    # top_k = lambda position, top: np.argsort([vector_distance(position, p) for p in reachable_positions])[:top]

    # initial position and rotation
    cur_pos = controller.last_event.metadata['agent']['position']
    cur_rot = controller.last_event.metadata['agent']['rotation']
    cur_obs = controller.last_event.frame
    # cur_depth = controller.last_event.depth_frame
    # cur_seg = controller.last_event.semantic_segmentation_frame
    # cur_ins = controller.last_event.instance_segmentation_frame
    rgb_writer.append(cur_obs)
    # rgb_writer.append_data(cur_obs)
    # seg_writer.append_data(cur_seg)
    # ins_writer.append_data(cur_ins)
    # depth_writer = np.vstack([depth_writer, cur_depth[None,:]])
    all_data = {
        'position': [cur_pos],
        'rotation': [cur_rot],
        'action': [],
        'action_indices': [],
        'objects': [],
    }

    step_count, turn_count, default_turn = 0, 0, 1
    action_queue = [action_space[0]] * max_step * 10
    while step_count < max_step:
        action = action_queue.pop()
        event = controller.step(action=action)
        if not event.metadata['lastActionSuccess']:
            while action_queue.pop() != action_space[0]: pass
            action_queue.extend([action_space[default_turn]]* np.random.choice(range(1,4),p=[0.5,0.3,0.2]))
            turn_count += 1
            if turn_count % change_turn == 0:
                turn_count = 0
                default_turn = 3 - default_turn
            # action_queue = [action_space[default_turn]] * np.random.choice(5,p=[0.4,0.2,0.2,0.1,0.1]) + action_queue
            continue
        
        event = controller.last_event  # Get the latest event
        cur_pos = event.metadata['agent']['position']
        cur_rot = event.metadata['agent']['rotation']
        all_data['position'].append(cur_pos)
        all_data['rotation'].append(cur_rot)
        all_data['action'].append(action)
        all_data['action_indices'].append(action_space.index(action))
        for o in event.metadata['objects']:
            if o['distance'] < 1 and o['visible']:
                all_data['objects'].append(dict(
                    timestep=step_count,
                    position=o['position'],
                    rotation=o['rotation'],
                    distance=o['distance'],
                    objectId=o['objectId'],
                    objectType=o['objectType'],
                ))
                # all_object_id[o['objectId']].append(step_count)

        # Save the frames
        cur_obs = event.frame
        # cur_depth = event.depth_frame
        # cur_seg = event.semantic_segmentation_frame
        # cur_ins = event.instance_segmentation_frame
        # print(cur_obs.max(), cur_depth.max(), cur_seg.max(), cur_ins.max())
        rgb_writer.append(cur_obs)
        # seg_writer.append_data(cur_seg)
        # ins_writer.append_data(cur_ins)
        # depth_writer = depth_writer = np.vstack([depth_writer, cur_depth[None,:]])
        
        step_count += 1
        # if step_count % 100 == 0:
        #     num_object_id = len(set([o['objectId'] for o in all_data['objects']]))
        #     num_object_type = len(set([o['objectType'] for o in all_data['objects']]))
        #     print(f"Step: {step_count}/{max_step} | total {len(all_data['objects']):03d}, {num_object_id:02d} in {num_object_type:02d}")

    # Save all data to disk
    # all_data['all_object_type'] = all_object_type
    # all_data['object_to_color'] = o2c
    # all_data['color_to_object'] = c2o
    
    # all_data['goal_timestep'] = all_object_id
    # save_json(all_data, osp.join(data_dir, 'data.json'))

    # np.save(osp.join(data_dir, 'depth.npy'), depth_writer)
    controller.stop()#,rgb_writer.close(),seg_writer.close(),ins_writer.close()
    num_object_id = len(set([o['objectId'] for o in all_data['objects']]))
    num_object_type = len(set([o['objectType'] for o in all_data['objects']]))
    print(f'Finish collecting {max_step} steps, {num_object_id} objects, {num_object_type} object types in {scene_name}')

    return rgb_writer, all_data, (num_object_type, num_object_id)  # sort by the num_object_type first and then num_object_id


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
        plt.show()

    return pred_actions, pred_action_indices, [thre_x, thre_y], accuracy


def check_and_decode_action_from_video(save_dir):
    # check 
    all_data = read_json(osp.join(save_dir, 'data.json'))
    print(all_data.keys(),len(all_data['position']),len(all_data['rotation']),len(all_data['action']),len(all_data['action_indices']),len(all_data['objects']))
    if osp.exists(osp.join(save_dir, 'depth.npy')): 
        depths = np.load(osp.join(save_dir, 'depth.npy'))
        print(depths.shape,depths.max(),depths.min())
    
    # Decode flow action
    action_indices = np.array(all_data['action_indices'])
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,4)
    for i in range(3):
        point_group = points[action_indices==i].T
        print(i, [f'{each:.2f}' for each in sorted(point_group[1])])
        print(i, [f'{each:.2f}' for each in sorted(point_group[1], reverse=True)])
        ax[i].scatter(*point_group)
        ax[i].set_xlim(-300,300)
        ax[i].set_ylim(-150,350)
    ax[3].scatter(*points.T,c=action_indices)
    plt.show()
    '''
    video_frames = mp42np(osp.join(save_dir, 'rgb_video.mp4'))
    print(video_frames.shape)
    flowActionDecoder = Flow2Action()
    if osp.exists(osp.join(save_dir, 'points.npy')):
        points = np.load(osp.join(save_dir, 'points.npy'))
        pred_actions, pred_action_indices, points = flowActionDecoder.decode_action_from_vec(points)
    else:
        pred_actions, pred_action_indices, flow = flowActionDecoder.predict_actions_from_one_video(video_frames)
        # visualize the flow, optional
        # trunk = 10
        # for i,(origin_image,flow_output,gt_action,pred_action) in enumerate(zip(video_frames,flow,all_data['action'],pred_actions)):
        #     show(origin_image,save=os.path.join(osp.join(save_dir, 'vis'),f'origin_{i}.jpg'),display=False)
        #     draw_flow(flow_output, os.path.join(osp.join(save_dir, 'vis'),f'vis_flow_{i}.jpg'),tag=f'Step:{i} | GT: {gt_action}  Pred: {pred_action}')
        #     if i > trunk: break
        
        points = reduce(flow, 'b h w c -> b c', 'mean')
        np.save(osp.join(save_dir, 'points.npy'), points)
    
    assert len(pred_actions) == len(all_data['action'])
    correctness = np.array([a == b for a, b in zip(pred_action_indices, action_indices)])
    for i in np.where(correctness==0)[0]:
        print(f'GT: {action_indices[i-5:i+5]} | Pred: {pred_action_indices[i-5:i+5]}')
    print('acc:', np.mean(correctness))

    # save the decoded action
    all_data['pred_actions'] = pred_actions
    all_data['pred_action_indices'] = pred_action_indices
    save_json(all_data, osp.join(save_dir, 'data.json'))

def setup_controller(scene_name):
    # Initialize the controller with the desired configuration
    controller = Controller(
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_step', type=int, default=900)  # 1800 -> 1 min
    parser.add_argument('--fct', type=int, default=10)
    parser.add_argument('--superglue', action="store_true")
    args = parser.parse_args()
    seed_everything(42)

    if args.superglue:
        action_decoder = SuperGlue2Action()
        for scene_name, category in [(f'FloorPlan_Train{i}_{j}', 'train') for i in range(1,13) for j in range(1,5)] + [(f'FloorPlan_Train{i}_{5}', 'val') for i in range(1,13)] + [(f'FloorPlan_Val{i}_{j}', 'val') for i in range(1,4) for j in range(1,6)]:
            data_dir=f'offline-dataset/robothor-dataset/900/{category}/{scene_name}'
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
            for thre_x in range(10, 61, 1):
                for thre_y in range(10, 61, 1):
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


    for num_total, category in [(12*5, 'Train'), (3*5, 'Val')]:
        for j in tqdm(range(0,num_total),total=num_total):
            beg = time.time()
            layout_type, layout_num = j//5+1, j%5+1
            scene_name = f'FloorPlan_{category}{layout_type}_{layout_num}'
            split = 'train' if layout_num < 5 and category == 'Train' else 'val'
            data_dir=f'offline-dataset/robothor-dataset/900/{split}/{scene_name}'
            if not osp.exists(data_dir):
                os.makedirs(data_dir)

            # step 1: collect the most diverse data and record 
            if not osp.exists(osp.join(data_dir, 'data.json')):
                num_rollout = 5
                samples = [rollout(scene_name=scene_name, max_step=args.max_step,change_turn=args.fct*i) for i in range(1, 1+num_rollout)]
                rgb_writer, all_data, (num_object_type, num_object_id) = sorted(samples, key=lambda x: x[-1])[-1]

                all_data['num_object_type'] = num_object_type
                all_data['num_object_id'] = num_object_id
                
                # save the video 
                imageio.mimwrite(osp.join(data_dir, 'rgb_video.mp4'), rgb_writer, fps=10)
                print(f'Collected {num_object_type} object types, {num_object_id} objects in {scene_name}')
            else:
                rgb_writer = mp42np(osp.join(data_dir, 'rgb_video.mp4'))
                all_data = read_json(osp.join(data_dir, 'data.json'))


            # step 2: tune the threshold for psuedo action labeling
            if 'threshold' not in all_data:
                flowActionDecoder = Flow2Action()
                action_indices = np.array(all_data['action_indices'])
                xrange = range(10, 91, 1)
                yrange = range(10, 41, 1)
                samples = [append_psuedo_action_labeling(flowActionDecoder, action_indices, data_dir, thre_x=thre_x, thre_y=thre_y) for thre_x in xrange for thre_y in yrange]
                pred_actions, pred_action_indices, threshold, accuracy = sorted(samples, key=lambda x: x[-1])[-1]
                print(f'Choose threshold: {threshold} | Accuracy: {accuracy}')
                
                # save the decoded action and data
                all_data['pred_actions'] = pred_actions
                all_data['pred_action_indices'] = pred_action_indices
                all_data['threshold'] = threshold
                all_data['accuracy'] = accuracy
                save_json(all_data, os.path.join(data_dir, 'data.json'))


            # step 3: make testing points
            if split != 'val': continue
            controller = setup_controller(scene_name)
            goal_dir = os.path.join(data_dir, 'goal_frames')
            os.makedirs(goal_dir, exist_ok=True)
            positions = all_data['position']
            rotations = all_data['rotation']
            out_episodes = []
            count_dict = {obj['objectType']:0 for obj in all_data['objects']}
            N_EACH, DIS_THRE = 5, 2
            repeat = 0
            while repeat < 2:
                for o in all_data['objects']:
                    timestep, objtype, objid = o['timestep'], o['objectType'], o['objectId'] 
                    target_position = positions[timestep]
                    goal_image_name = f'{scene_name}_{objtype}_{timestep}.png'
                    extent = 100
                    t = np.random.choice(np.arange(max(timestep-extent,0),min(timestep+extent,len(rgb_writer)-1)))
                    # t = np.random.choice(len(videos)-2)
                    initial_position = positions[t]
                    initial_rotation = rotations[t]
                    initial_horizon = 0.0
                    euclidean_distance = vector_distance(initial_position, target_position)
                    try:
                        shortest_path = get_shortest_path_to_object(controller, objid, initial_position, initial_rotation)
                    except ValueError:
                        shortest_path = []
                    if shortest_path and euclidean_distance > DIS_THRE and count_dict[objtype] < N_EACH:
                        shortest_path_length = path_distance(shortest_path)
                        # save episode
                        out_episodes.append({
                            "id": f'{scene_name}_{objtype}_{count_dict[objtype]}',
                            "scene": scene_name,
                            "object_type": objtype,
                            "object_id": objid,
                            "initial_position": initial_position,
                            "initial_orientation": initial_rotation['y'],
                            "initial_horizon": initial_horizon,
                            "start_timestep": int(t),
                            "goal_image_name":goal_image_name,
                            "target_position": target_position,
                            "euclidean_distance": euclidean_distance,
                            "shortest_path_length": shortest_path_length,
                            "shortest_path": shortest_path,
                        })
                        count_dict[objtype] += 1
                        imageio.imwrite(os.path.join(goal_dir,goal_image_name),rgb_writer[timestep])   # save goal image
                repeat += all([v == N_EACH or v == 0 for v in count_dict.values()])

            out_episodes = sorted(out_episodes,key=lambda x:x['id'])
            with gzip.open(os.path.join(data_dir, f'{scene_name}_datasets.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(out_episodes, f, indent=4)
            print(f'{len(out_episodes)} testing episodes have been constructed in {data_dir}!')
            print(f'{scene_name} done in {time.time()-beg:.2f} s')

