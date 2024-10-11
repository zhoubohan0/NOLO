import os, sys, torch, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json, random
import os.path as osp
import pdb
from glob import glob
import matplotlib.pyplot as plt
import imageio
import numpy as np
from ai2thor.controller import Controller
from ai2thor.util.metrics import (
    get_shortest_path_to_object,
    get_shortest_path_to_object_type,
    vector_distance,
)
from ai2thor_colab import *
from PIL import Image

from gmflow.flow2action import Flow2Action, draw_flow, reduce, show

action_space =  ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]


def decode_action_from_pose(rotation, threshold=10):
    '''
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

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def mp42np(mp4_file, way='ffmpeg'):
    if way == 'decord':
        return np.stack([im.asnumpy() for im in decord.VideoReader(mp4_file, num_threads=16)])
    elif way == 'ffmpeg':
        return np.stack([im for im in imageio.get_reader(mp4_file,  'ffmpeg')])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def collect_video_in_environment(data_dir,max_step=900,change_turn=50):
    '''
    input: None
    output: data_dir/rgb_video.mp4, data_dir/data.json, data_dir/points.npy (middle result)
    '''
    rgb_writer = imageio.get_writer(osp.join(data_dir, 'rgb_video.mp4'), fps=10)
    # depth_writer = np.empty((0, 480, 640), dtype=np.float32)
    # seg_writer = imageio.get_writer(osp.join(data_dir, 'semantic_video.mp4'))
    # ins_writer = imageio.get_writer(osp.join(data_dir, 'instance_video.mp4'))

    # Initialize the controller with the desired configuration
    controller = Controller(
        width=640,
        height=480,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        continuousMode=True,
        snapToGrid=False,
        headless=False,  
        agentMode="locobot",
        save_image_per_frame=True,
        image_dir=data_dir,
    )
    controller.reset(
        scene=osp.basename(data_dir),
        renderObjectImage=True,
        renderClassImage=True,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        renderFlowImage=True,    
    )
    # controller.step(action="RandomizeMaterials",useTrainMaterials=True, useValMaterials=True)

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
    rgb_writer.append_data(cur_obs)
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
        rgb_writer.append_data(cur_obs)
        # seg_writer.append_data(cur_seg)
        # ins_writer.append_data(cur_ins)
        # depth_writer = depth_writer = np.vstack([depth_writer, cur_depth[None,:]])
        
        step_count += 1
        if step_count % 100 == 0:
            print(f"Step: {step_count}/{max_step} | total {len(all_data['objects']):03d}, {len(set([o['objectId'] for o in all_data['objects']])):02d} in {len(set([o['objectType'] for o in all_data['objects']])):02d}")

    # Save all data to disk
    # all_data['all_object_type'] = all_object_type
    # all_data['object_to_color'] = o2c
    # all_data['color_to_object'] = c2o
    
    # all_data['goal_timestep'] = all_object_id
    save_json(all_data, osp.join(data_dir, 'data.json'))
    # np.save(osp.join(data_dir, 'depth.npy'), depth_writer)
    controller.stop(),rgb_writer.close()#,seg_writer.close(),ins_writer.close()


def append_psuedo_action_labeling(data_dir, thre_x=90,thre_y=35):
    # check_and_decode_action_from_video(data_dir)
    flowActionDecoder = Flow2Action()
    if os.path.exists(os.path.join(data_dir, 'points.npy')):
        points = np.load(os.path.join(data_dir, 'points.npy'))
        pred_actions, pred_action_indices, points = flowActionDecoder.decode_action_from_vec(points,thre_x, thre_y)
    else:
        video_frames = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
        pred_actions, pred_action_indices, flow = flowActionDecoder.predict_actions_from_one_video(video_frames,thre_x, thre_y)
        points = reduce(flow, 'b h w c -> b c', 'mean')
        np.save(os.path.join(data_dir, 'points.npy'), points)
    
    all_data = read_json(os.path.join(data_dir, 'data.json'))
    action_indices = np.array(all_data['action_indices'])
    assert len(pred_actions) == len(action_indices)
    correctness = np.array([a == b for a, b in zip(pred_action_indices, action_indices)])
    accuracy = np.mean(correctness)
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

    # save the decoded action
    all_data['pred_actions'] = pred_actions
    all_data['pred_action_indices'] = pred_action_indices
    all_data['threshold'] = [thre_x, thre_y] 
    save_json(all_data, os.path.join(data_dir, 'data.json'))


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--scene_name', type=str, default='1_2')
    parser.add_argument('--max_step', type=int, default=900)  # 1800 -> 1 min
    parser.add_argument('-x', '--thre_x', type=int, default=50)
    parser.add_argument('-y', '--thre_y', type=int, default=20)
    parser.add_argument('--fct', type=int, default=30)
    args = parser.parse_args()

    seed_everything(42)
    data_dir=f'offline-dataset/robothor-dataset/900/FloorPlan_Train{args.scene_name}'
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    # collect_video_in_environment(data_dir, max_step=args.max_step,change_turn=args.fct)
    append_psuedo_action_labeling(data_dir, thre_x=args.thre_x, thre_y=args.thre_y)
