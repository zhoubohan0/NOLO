import os, argparse
import cv2, imageio
import numpy as np
import time
import threading
import pyrealsense2 as rs
import asyncio
import torch
import torch.nn.functional as F
import imageio.v2 as iio
import math
from collections import defaultdict
from glob import glob
from torchvision.transforms import Compose, RandomResizedCrop, ToPILImage
from recbert_policy.train_vnbert_real import load_SA
from recbert_policy.vnbert import VNBERTPolicy
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactortyInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient, PathPoint, SPORT_PATH_POINT_SIZE

def print_color(msg, color="green"):
    if color == "red":
        print("\033[31m{}\033[0m".format(msg))
    elif color == "green":
        print("\033[32m{}\033[0m".format(msg))
    elif color == "yellow":
        print("\033[33m{}\033[0m".format(msg))
    elif color == "blue":
        print("\033[34m{}\033[0m".format(msg))
    elif color == "purple":
        print("\033[35m{}\033[0m".format(msg))
    elif color == "cyan":
        print("\033[36m{}\033[0m".format(msg))
    elif color == "gray":
        print("\033[37m{}\033[0m".format(msg))
    else:
        print(msg)

def rearrange(items):
    category_dict = defaultdict(list)
    for item in items:
        category = os.path.basename(item).split('_')[0]  # Extract the category before '_'
        category_dict[category].append(item)
    for category in category_dict:
        np.random.shuffle(category_dict[category])
    result = []
    while category_dict:
        removed_categories = []
        for category in list(category_dict):
            if category_dict[category]:
                result.append(category_dict[category].pop(0))  # Take the first element from the category list
            if not category_dict[category]:  # If the category list is empty, mark it for removal
                removed_categories.append(category)

        # Remove categories that are out of elements
        for category in removed_categories:
            del category_dict[category]

    return result

class Policy(VNBERTPolicy):
    def __init__(self, dataset, ckpt_file='', mode='A',context_type='SA',**kwargs):
        super(Policy, self).__init__()
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_type = context_type
        self.action_threshold = 0.5
        self.video_frames, self.actions = dataset
        self.num_context_sample = len(self.actions) - 1
        self.actions = torch.LongTensor(self.actions)
        self.augment = Compose([
            ToPILImage(),
            RandomResizedCrop(224, scale=(0.99, 1.0), ratio=(0.99, 1.01)),
        ])
        if ckpt_file:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            ckpt = {k.replace('module.','') if 'module.' in k else k:v for k,v in ckpt.items()}
            self.load_state_dict(ckpt,strict=True)
            print(f'Load checkpoint from {ckpt_file}')
        
    def reset(self,):
        # pre-compute fixed context embeddings using trained state encoder
        with torch.no_grad():
            if 'S' in self.context_type:
                self.context_frames = torch.from_numpy(np.stack([np.array(self.augment(frame)) for frame in self.video_frames])).to(self.device)
                self.context_actions = self.actions.to(self.device) if self.context_type == 'SA' else None
                self.st, enc_context, _ = self.enc_context(self.context_frames, self.context_actions)  # (1, Dc), (1, Tc+1, Dc)
                self.enc_pure_context = enc_context[:,1:,:]  # (1, Tc, Dc)
            elif self.context_type == 'None':
                self.st = torch.zeros(1, 1, self.hidden_size).to(self.device)
                self.enc_pure_context = torch.zeros(1, self.num_context_sample, self.hidden_size).to(self.device)
        self.goal_emb = None
        self.stored_actions = dict(last_action=-1,duration=-1,action_prob=0)

    def set_goal(self, goal_frame):
        self.goal_emb = self.goal_encoder(torch.from_numpy(np.array(self.augment(goal_frame))).to(self.device).unsqueeze(0))

    def act(self, rgb_frame):
        assert self.goal_emb is not None
        
        if self.stored_actions['duration'] == -1:  # generate a new action
            with torch.no_grad():
                enc_context = torch.cat((self.st, self.enc_pure_context), dim=1)
                self.st, logit = self.enc_step(enc_context, self.goal_emb, torch.from_numpy(np.array(self.augment(rgb_frame))).to(self.device).unsqueeze(0))
                q_value = self.critic_head(self.st).squeeze(1)          # (1, N)
                action_dist = F.log_softmax(logit,-1).exp()             # (1, N)
                self.keep_mode = action_dist.shape[-1] > 3
            if self.mode == 'Q':
                if np.random.uniform(0,1) > 0.001:
                    with torch.no_grad():
                        action_dist = (action_dist / action_dist.max(-1, keepdim=True)[0] > self.action_threshold).float()
                        pred_action_indice = int((action_dist * q_value + (1. - action_dist) * -1e8).argmax(-1))
                else:
                    pred_action_indice = np.random.choice(action_dist.shape[-1])
            elif 'A' in self.mode:
                pred_action_indice = np.random.choice(range(action_dist.shape[-1]), p=action_dist.squeeze().detach().cpu().numpy())
            
            self.stored_actions['last_action'] = pred_action_indice if self.keep_mode == False else pred_action_indice // 3 
            self.stored_actions['duration'] = 0 if self.keep_mode == False else pred_action_indice % 3
        actual_action = self.stored_actions['last_action']  
        self.stored_actions['duration'] -= 1
        return actual_action               
 


class HighLevelController:
    def HighStateHandler(self, msg: SportModeState_):
        self.robot_state = msg
    
    def run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
    async def start_processing(self):
        while True:
            await self.process_realsense()

    async def process_realsense(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                await asyncio.sleep(0.001)  # Reduce sleep time for higher frequency
                continue

            # Convert images to numpy arrays
            # depth_image = np.expand_dims(np.asanyarray(depth_frame.get_data()), axis=2)
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # # Get camera extrinsics
            # pose_matrix = rs.video_stream_profile(color_frame.profile).get_extrinsics_to(rs.video_stream_profile(depth_frame.profile))
            # rotation = np.asanyarray(pose_matrix.rotation)
            # translation = np.asanyarray(pose_matrix.translation)
            with self.lock:
                self.rgb = color_image
                self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
                self.dpt = depth_image * 0.001  # Convert depth to meters

            await asyncio.sleep(0.001)  # Reduce sleep time for higher frequency

    def __init__(self, forward_distance=0.3, turn_angle=np.pi/6, record_file=''):
        self.forward_distance, self.turn_angle = forward_distance, turn_angle
        self.action_names = ['forward', 'turn_left', 'turn_right', 'backward']
        self.rgb_writer = iio.get_writer(record_file, fps=10) if record_file else None
        self.act_file = record_file.replace('.mp4', '.npy') if record_file else ''
        self.act_record = []
        # Initialize lock for thread-safe operations
        self.lock = threading.Lock()

        # to depth camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config) # Start streaming
        self.rgb, self.dpt = np.zeros([640, 480]), np.zeros([640, 480])

        self.robot_state = unitree_go_msg_dds__SportModeState_()
        
        ChannelFactortyInitialize(0, "eth0")
        sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sport_sub.Init(self.HighStateHandler, 10)
        
        self.dt = 0.01
        self.client = SportClient()  # Create a sport client
        self.client.SetTimeout(10.0)
        self.client.Init()

        # get initial state
        self.px0 = self.robot_state.position[0]
        self.py0 = self.robot_state.position[1]
        self.yaw0 = self.robot_state.imu_state.rpy[2]

        self.xyz = self.observe()['position']

        # Create and run the event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_loop, args=(self.loop,))
        self.thread.start()
        # Schedule the async tasks
        asyncio.run_coroutine_threadsafe(self.start_processing(), self.loop)

    '''core functions'''
    def move(self, vx, vy, vyaw, log=False):
        '''
        move with velocity vx, vy, vyaw
        '''
        with self.lock:
            elapsed_time, start_time = 1.0, time.time()
            for i in range(int(elapsed_time / self.dt)):
                self.client.Move(vx, vy, vyaw)
                time.sleep(self.dt)
            self.client.StopMove()
            if log: 
                end_time = time.time()
                print_color(f"acted {vx, vy, vyaw}, for {end_time-start_time} s")

    def forward(self, log=False):
        self.move(self.forward_distance, 0, 0, log)

    def backward(self, log=False):
        self.move(-self.forward_distance, 0, 0, log)

    def turn_left(self, log=False):
        self.move(0, 0, self.turn_angle, log)

    def turn_right(self, log=False):
        self.move(0, 0, -self.turn_angle, log)
    
    def stop(self):
        self.client.StopMove()

    def observe(self):
        """
        Get position, rpy, rgb, depth
        """
        positions = []
        rpys = []
        for _ in range(10):
            position, rpy = self.robot_state.position, self.robot_state.imu_state.rpy
            positions.append(position)
            rpys.append(rpy)
            time.sleep(0.01)  # 适当的延迟以获取不同的传感器读数
        avg_position = np.mean(positions, axis=0)
        avg_rpy = np.mean(rpys, axis=0)
        # with self.lock:
        #     return self.rgb.copy(), self.dpt.copy()
        return {
            "position": avg_position.tolist(), 
            "rpy": avg_rpy.tolist(),
            "rgb": self.rgb,
            "depth": self.dpt,
        }
    
    def step(self, action_id):
        '''
        action_id: [0, 1, 2]
        action: ['forward', 'turn_left', 'turn_right']
        '''
        action = self.action_names[action_id]
        eval(f"self.{action}()")
        next_obs = self.observe()
        return next_obs

    def start_single_nav_task(self, data_dir, goal_dir, ckpt_file):
        goal_image_files = glob(os.path.join(goal_dir, '*_*.png'))
        goal_image_files = rearrange(goal_image_files)
        np.random.shuffle(goal_image_files)
        goal_images = [imageio.v2.imread(f) for f in goal_image_files]
        # load dataset and initialize the agent
        agent = Policy(load_SA(data_dir), ckpt_file)
        agent.to(agent.device).reset()
        # iter all goal images and act till completion
        print(f'Agent is reset! Begin to navigate.')
        for igoal, (goal_file, goal_image) in enumerate(zip(goal_image_files, goal_images)):
            print(f'{igoal}th goal: {goal_file}')
            agent.set_goal(goal_image)
            obs = self.observe()['rgb']
            for t in range(200):
                action = agent.act(obs)
                print(f'timestep: {t} | action:{action}')
                #action = np.random.choice(3)
                obs = self.step(action)['rgb']

    '''auxiliary functions'''    
    def standdown(self):
        self.client.StandDown()
        
    def hello(self):
        self.client.Hello()

    def switch_gait(self, gait):
        self.client.SwitchGait(gait)

    def act_point(self, action, c_xy_yaw):
        # print(self.px0, self.py0, self.yaw0)
        # print(c_xy_yaw)
        # time_seg = 0.2
        # time_temp = self.t - time_seg
        # path = []
        # for i in range(30):
        #     time_temp += time_seg

        #     px_local = 0.5 * math.sin(0.5 * time_temp)
        #     py_local = 0
        #     yaw_local = 0
        #     vx_local = 0.25 * math.cos(0.5 * time_temp)
        #     vy_local = 0
        #     vyaw_local = 0

        #     path_point_tmp = PathPoint(0, 0, 0, 0, 0, 0, 0)

        #     path_point_tmp.timeFromStart = i * time_seg
        #     path_point_tmp.x = (
        #         px_local * math.cos(self.yaw0)
        #         - py_local * math.sin(self.yaw0)
        #         + self.px0
        #     )
        #     path_point_tmp.y = (
        #         px_local * math.sin(self.yaw0)
        #         + py_local * math.cos(self.yaw0)
        #         + self.py0
        #     )
        #     path_point_tmp.yaw = yaw_local + self.yaw0
        #     path_point_tmp.vx = vx_local * math.cos(self.yaw0) - vy_local * math.sin(
        #         self.yaw0
        #     )
        #     path_point_tmp.vy = vx_local * math.sin(self.yaw0) + vy_local * math.cos(
        #         self.yaw0
        #     )
        #     path_point_tmp.vyaw = vyaw_local

        #     path.append(path_point_tmp)

        # self.client.TrajectoryFollow(path)
        # print(asd)

        # 提取 action 的 x 和 y 坐标，以及角度
        t = 0
        dt = 0.01
        # for _ in range(600):
        t += dt
        xg, yg, angle = action

        time_seg = 0.2
        # time_temp = self.t - time_seg
        path = []
        cx, cy, cyaw = c_xy_yaw

        # # 计算当前航向角度和目标点之间的角度
        # target_angle = math.atan2(yg - cy, xg - cx)

        # 计算当前角度和目标角度之间的差值
        angle_difference = (angle - cyaw) % (2 * np.pi)

        # print(angle)
        # print(asd)
        # print(angle_difference)
        # 将角度差值规范化到 [-π, π] 范围内
        if angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        elif angle_difference < -math.pi:
            angle_difference += 2 * math.pi
        # print(angle_difference)
        # # 判断转向方向
        # if angle_difference > 0:
        #     turn_direction = "right"
        # else:
        #     turn_direction = "left"

        # print(turn_direction)
        for i in range(30):  # 创建一个简单的路径，共30个点
            # time_temp += time_seg

            # 插值计算路径点
            px_local = cx + (xg - cx) * (i + 1) / 30
            py_local = cy + (yg - cy) * (i + 1) / 30

            # if turn_direction == "right":
            #     # 如果向右转，yaw_local 需要逐渐增加
            #     yaw_local = cyaw + angle_difference * (i + 1) / 30
            # else:
            #     # 如果向左转，yaw_local 需要逐渐减少
            #     yaw_local = cyaw - angle_difference * (i + 1) / 30
            yaw_local = cyaw + angle_difference * (i + 1) / 30

            vx_local = (xg - cx) / (30 * time_seg)
            vy_local = (yg - cy) / (30 * time_seg)
            vyaw_local = angle_difference / (30 * time_seg)

            path_point_tmp = PathPoint(0, 0, 0, 0, 0, 0, 0)

            path_point_tmp.timeFromStart = i * time_seg
            path_point_tmp.x = px_local
            path_point_tmp.y = py_local
            path_point_tmp.yaw = yaw_local
            path_point_tmp.vx = vx_local
            path_point_tmp.vy = vy_local
            path_point_tmp.vyaw = vyaw_local

            path.append(path_point_tmp)

        self.client.TrajectoryFollow(path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='offline-dataset/maze-dataset/0')
    parser.add_argument('--ckpt_file', '-c', type=str, default='logs/maze_5+1traj/policy_22000.pth')
    args = parser.parse_args()
    args.goal_dir = os.path.join(args.data_dir, 'goal_frames')
    hlc = HighLevelController()
    hlc.start_single_nav_task(args.data_dir, args.goal_dir, args.ckpt_file)
