import os.path as osp
import numpy as np
import imageio, json, os
from tqdm import tqdm
import habitat_sim
dummy_path = habitat_sim.ShortestPath()
from scripts.collect_habitat_all import get_sim_and_action_config, mp42np, read_json


def get_path_from_start_to_end(sim, start, end):
    dummy_path.requested_start = start
    dummy_path.requested_end = end
    sim.pathfinder.find_path(dummy_path)
    return [p.astype(float).tolist() for p in dummy_path.points], dummy_path.geodesic_distance


def standard_generate_data(data_dir, N_EACH=20):
    goal_dir = osp.join(data_dir, 'goal_frames')
    os.makedirs(goal_dir, exist_ok=True)
    videos = mp42np(osp.join(data_dir, 'rgb_video.mp4'))
    all_data = read_json(osp.join(data_dir,'data.json'))
    positions = all_data['position']
    rotations = all_data['rotation']

    out_episodes = []
    count_dict = {obj['objectType']:0 for obj in all_data['objects']}
    N_CATEGORY = len(count_dict)
    
    while True:
        for o in tqdm(all_data['objects'],total=len(all_data['objects'])):
            timestep, objtype, objid = o['timestep'], o['objectType'], o['objectId']
            target_position = positions[timestep]
            goal_image_name = f'{scene_name}_{objtype}_{timestep}.png'
            extent = 100
            t = np.random.choice(np.arange(max(timestep-extent,0),min(timestep+extent,len(videos)-2)))
            initial_position = positions[t]
            initial_rotation = rotations[t]
            shortest_path, shortest_path_length = get_path_from_start_to_end(sim, initial_position, target_position)
            if shortest_path and shortest_path_length >= 1.0 and shortest_path_length<=6.0 and count_dict[objtype] < N_EACH and np.random.uniform() < 0.5:
                out_episodes.append({
                    "id": f'{scene_name}_{objtype}',
                    "scene": scene_name,
                    "object_type": objtype,
                    "object_id": objid,
                    "initial_position": initial_position,
                    "initial_orientation": initial_rotation,
                    "goal_image_name":goal_image_name,
                    "target_position": target_position,
                    "shortest_path_length": shortest_path_length,
                    "shortest_path": shortest_path,
                })
                imageio.imwrite(osp.join(goal_dir,goal_image_name),videos[timestep])  # save goal image
                count_dict[objtype] += 1
                if all([v == N_EACH for v in count_dict.values()]):
                    return sorted(out_episodes,key=lambda x:x['id'])
        print(count_dict)

if __name__ == "__main__":
    sim, cfg = get_sim_and_action_config()
    scene_name = osp.basename(osp.dirname(cfg.sim_cfg.scene_id))
    data_dir = f'/home/zbh/Desktop/zbh/Proj/nav/my/oso/offline-dataset/habitat-dataset/900/{scene_name}'
    out_episodes = standard_generate_data(data_dir)

    with open(osp.join(data_dir, f'{scene_name}_datasets.json'), 'w') as f:
        json.dump(out_episodes, f, indent=4)
    print(f'{len(out_episodes)} episodes have been constructed!')
