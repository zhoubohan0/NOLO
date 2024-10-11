import json, gzip, imageio, random, torch, pickle, argparse
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from ai2thor.util.metrics import get_shortest_path_to_object_type, get_shortest_path_to_object,vector_distance,path_distance
from ai2thor.controller import Controller


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


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

def old_generate_data():
    controller = Controller(
        width=640,
        height=480,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        continuousMode=True,
        snapToGrid=False,
        headless=True,
        save_image_per_frame = False,
        agentMode="locobot",
    )
    all_objects = controller.last_event.metadata['objects']
    video_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/dataset/videos'
    input_json_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/allenact/datasets/robothor-pointnav/val/episodes'
    output_json_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/robothor-imagenav/dataset/try/episodes'
    index_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/dataset/segvideos'

    def get_res_until_succeed(controller,available_objectIds,pose):
        success = False
        i = 0
        episode = {}
        while not success:
            try:
                episode = get_shortest_path_to_object(controller,available_objectIds[i],**pose)
                success = True
            except ValueError:
                i += 1
                if i == len(available_objectIds):
                    break
        return episode

    for input_json_file in tqdm(glob(os.path.join(input_json_dir, '*.json'))):
        with open(input_json_file, 'r') as f:
            episodes = json.load(f)
            out_episodes = []
        for i in tqdm(range(len(episodes)),total=len(episodes)):
            id = episodes[i]['id']
            pkl_file = os.path.join(index_dir, id+'.pkl')
            if os.path.exists(pkl_file):
                video_file = os.path.join(video_dir, id+'.mp4')
                frames = [im for im in imageio.get_reader(video_file,'ffmpeg')]
                with open(os.path.join(index_dir, id+'.pkl'), 'rb') as pf:
                    index_dict = pickle.load(pf)[-1]
                    for object_type, timestep in index_dict.items():
                        episode = deepcopy(episodes[i])
                        episode['id']+=f'_{object_type}'
                        episode['object_type'] = object_type
                        episode['initial_horizon'] = 30
                        imageio.imwrite(os.path.join(video_dir.replace('videos','goal_frames'),episode['id']+'.png'),frames[timestep])# episode['goal_frame'] = base64.b64encode(frames[timestep].tobytes()).decode()

                        pose = dict(
                            initial_position={
                                "x": episode["initial_position"]["x"],
                                "y": episode["initial_position"]["y"],
                                "z": episode["initial_position"]["z"],
                            },
                            initial_rotation={
                                "x": 0,
                                "y": episode['initial_orientation'],
                                "z": 0
                            })
                        try:
                            shortest_path = get_shortest_path_to_object_type(controller,episode["object_type"],**pose)
                        except ValueError: # choose get_shortest_path_to_object
                            shortest_path = {}
                            # available_objectIds = [obj['objectId'] for obj in all_objects if obj['objectType'] == episode["object_type"]]
                            # shortest_path = get_res_until_succeed(controller,available_objectIds,pose)
                        finally:
                            episode['shortest_path'] = shortest_path
                        if shortest_path:
                            out_episodes.append(episode)
        with gzip.open(os.path.join(output_json_dir, os.path.basename(input_json_file).replace('.json', '.json.gz')), 'wt', encoding='utf-8') as f:
            json.dump(out_episodes, f, indent=4)
    
def new_generate_data(data_dir):
    scene_name = os.path.basename(data_dir)
    controller = Controller(
        width=640,
        height=480,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        continuousMode=True,
        snapToGrid=False,
        headless=True,  
        agentMode="locobot",
        agentType="stochastic",
        commit_id='bad5bc2b250615cb766ffb45d455c211329af17e',
        fieldOfView=63.453048374758716,
    )
    controller.reset(
        scene=scene_name,
        # renderObjectImage=True,
        # renderClassImage=True,
        # renderDepthImage=True,
        # renderInstanceSegmentation=True,
        # renderSemanticSegmentation=True,
        # renderFlowImage=True,    
    )

    
    videos = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
    all_data = read_json(os.path.join(data_dir,'data.json'))
    positions = all_data['position']
    rotations = all_data['rotation']
    goal_timesteps = all_data['goal_timestep']

    out_episodes = []
    for objid, timesteps in tqdm(goal_timesteps.items(),total=len(goal_timesteps)):
        objtype = objid.split('|')[0]
        sampled_timesteps = timesteps
        for timestep in sampled_timesteps:
            # save goal frame
            target_position = positions[timestep]
            goal_image_name = f'{scene_name}_{objtype}_{timestep}.png'
            for difficuly in ['easy', 'medium', 'hard']:
                if difficuly == 'easy':
                    t = max(timestep - np.random.choice(np.arange(10,20)),0)
                elif difficuly == 'medium':
                    t = max(timestep - np.random.choice(np.arange(20,40)),0)
                elif difficuly == 'hard':
                    t = max(timestep - np.random.choice(np.arange(40,50)),0)
                initial_position = positions[t]
                initial_rotation = rotations[t]
                initial_horizon = 0.0
                euclidean_distance = vector_distance(initial_position, target_position)
                try:
                    shortest_path = get_shortest_path_to_object(controller, objid, initial_position, initial_rotation)
                except ValueError:
                    shortest_path = []
                if shortest_path and euclidean_distance > 1:
                    shortest_path_length = path_distance(shortest_path)
                    # save episode
                    out_episodes.append({
                        "id": f'{scene_name}_{objtype}',
                        "scene": scene_name,
                        "object_type": objtype,
                        "object_id": objid,
                        "initial_position": initial_position,
                        "initial_orientation": initial_rotation['y'],
                        "initial_horizon": initial_horizon,
                        "goal_image_name":goal_image_name,
                        "target_position": target_position,
                        "euclidean_distance": euclidean_distance,
                        "shortest_path_length": shortest_path_length,
                        "shortest_path": shortest_path,
                    })
                    imageio.imwrite(os.path.join(os.path.dirname(os.path.dirname(data_dir)),'episodes','goal_frames',goal_image_name),videos[timestep])

    output_json_dir = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/robothor-imagenav/dataset/one_traj/episodes/'
    with gzip.open(os.path.join(output_json_dir, f'{scene_name}.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump(out_episodes, f, indent=4)
    print(f'{len(out_episodes)} episodes have been constructed!')


def standard_generate_data(data_dir, distance_threshold=1.5):
    scene_name = os.path.basename(data_dir)
    controller = Controller(
        width=640,
        height=480,
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        continuousMode=True,
        snapToGrid=False,
        headless=True,  
        agentMode="locobot",
        agentType="stochastic",
        commit_id='bad5bc2b250615cb766ffb45d455c211329af17e',
        fieldOfView=63.453048374758716,
    )
    controller.reset(scene=scene_name)

    goal_dir = os.path.join(data_dir, 'goal_frames')
    os.makedirs(goal_dir, exist_ok=True)

    videos = mp42np(os.path.join(data_dir, 'rgb_video.mp4'))
    all_data = read_json(os.path.join(data_dir,'data.json'))
    positions = all_data['position']
    rotations = all_data['rotation']

    out_episodes = []
    count_dict = {obj['objectType']:0 for obj in all_data['objects']}
    N_CATEGORY = len(count_dict)
    N_EACH = 20
    repeat = 0
    while repeat < 2:
        for o in all_data['objects']:
            timestep, objtype, objid = o['timestep'], o['objectType'], o['objectId'] 
            target_position = positions[timestep]
            goal_image_name = f'{scene_name}_{objtype}_{timestep}.png'
            extent = 100
            t = np.random.choice(np.arange(max(timestep-extent,0),min(timestep+extent,len(videos)-2)))
            # t = np.random.choice(len(videos)-2)
            initial_position = positions[t]
            initial_rotation = rotations[t]
            initial_horizon = 0.0
            euclidean_distance = vector_distance(initial_position, target_position)
            try:
                shortest_path = get_shortest_path_to_object(controller, objid, initial_position, initial_rotation)
            except ValueError:
                shortest_path = []
            if shortest_path and euclidean_distance > distance_threshold and count_dict[objtype] < N_EACH:
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
                imageio.imwrite(os.path.join(goal_dir,goal_image_name),videos[timestep])   # save goal image
        repeat += all([v == N_EACH or v == 0 for v in count_dict.values()])
        print(count_dict)
    return sorted(out_episodes,key=lambda x:x['id'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dirs = glob('offline-dataset/robothor-dataset/900/*/*')
    for data_dir in tqdm(data_dirs,total=len(data_dirs)):
        fsave = os.path.join(data_dir, f'{os.path.basename(data_dir)}_datasets.json.gz')
        if not os.path.exists(fsave):
            out_episodes = standard_generate_data(data_dir=data_dir)
            with gzip.open(fsave, 'wt', encoding='utf-8') as f:
                json.dump(out_episodes, f, indent=4)
            print(f'{len(out_episodes)} episodes have been constructed in {data_dir}!')