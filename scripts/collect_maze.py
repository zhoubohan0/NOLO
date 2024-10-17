import json
import cv2
import os, argparse, sys
import numpy as np
import time
import threading
import pyrealsense2 as rs
import asyncio
import logging
import imageio.v2 as iio
import math
from collections import deque
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.basic_utils import mp42np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactortyInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.go2.sport.sport_client import SportClient, PathPoint, SPORT_PATH_POINT_SIZE

# constants
MIN_VSPEED = 1e-7
TIME_OVER = 0.15   

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

class WirelessHighLevelController:
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
        self.act_file = os.path.join(os.path.dirname(record_file), 'data.json') if record_file else ''
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
        wireless_sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        wireless_sub.Init(self.wireless_signal_control, 10)
        
        self.dt = 0.01
        self.client = SportClient()  # Create a sport client
        self.client.SetTimeout(10.0)
        self.client.Init()

        # get initial state
        self.px0 = self.robot_state.position[0]
        self.py0 = self.robot_state.position[1]
        self.yaw0 = self.robot_state.imu_state.rpy[2]

        self.xyz = self.observe()['position']

        # reset signal recognizer 
        self.reset_signal_recognizer()

        # Create and run the event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_loop, args=(self.loop,))
        self.thread.start()
        # Schedule the async tasks
        asyncio.run_coroutine_threadsafe(self.start_processing(), self.loop)

        print('Start Remote control to collect trajectories!')

    '''core functions'''
    def move(self, vx, vy, vyaw, log=False):
        '''
        move with velocity vx, vy, vyaw
        '''
        elapsed_time, start_time = 1.0, time.time()
        for i in range(int(elapsed_time / self.dt)):
            self.client.Move(vx, vy, vyaw)
            time.sleep(self.dt)
        self.client.StopMove()
        end_time = time.time()
        if log: print_color(f"acted {vx, vy, vyaw}, for {end_time-start_time} s")

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
    
    def step(self, action):
        '''
        action: ['forward', 'turn_left', 'turn_right']
        '''
        eval(f"self.{action}()")
        next_obs = self.observe()
        return next_obs
    
    def step_act_seq(self, act_seq, record_file=''):
        '''
        act_seq: list of actions {0,1,2}
        '''
        rgb = self.observe()['rgb']
        if record_file:
            writer = iio.get_writer(record_file, fps=10)
            writer.append_data(rgb)
        for act_idx in act_seq:
            action = self.action_names[act_idx]
            assert action in self.action_names
            rgb = self.step(action)['rgb']
            writer.append_data(rgb)
        writer.close()

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


    '''wireless signals'''   
    def reset_signal_recognizer(self):
        self.key_state = [
            ["R1", 0],
            ["L1", 0],
            ["start", 0],
            ["select", 0],
            ["R2", 0],
            ["L2", 0],
            ["F1", 0],
            ["F2", 0],
            ["A", 0],
            ["B", 0],
            ["X", 0],
            ["Y", 0],
            ["up", 0],
            ["right", 0],
            ["down", 0],
            ["left", 0],
        ]
        self.state = False

    def wireless_signal_control(self, msg: WirelessController_):
        if msg.keys > 0:
            self.state = True
            for i in range(16):  #Update key state
                self.key_state[i][1] += (msg.keys & (1 << i)) >> i
        elif self.state and msg.keys == 0:
            # print(self.state, self.key_state)
            action = self.WirelessMove()
            next_obs = self.step(action)
            self.reset_signal_recognizer()
            print(f"step: {len(self.act_record)} | action:{action} | distance:{np.linalg.norm(np.array(self.xyz)-np.array(next_obs['position']))}")
            self.xyz = next_obs['position']
            if self.rgb_writer is not None:
                self.rgb_writer.append_data(next_obs['rgb'])
                self.act_record.append(self.action_names.index(action))
                if len(self.act_record) % 10 == 0:
                    # np.save(self.act_file, np.array(self.act_record))
                    with open(self.act_file, 'w') as f:
                        json.dump(dict(true_actions=self.act_record), f, indent=4)
                    print(f"Frames: {len(self.act_record)} | Saved to {self.act_file}")
    
    def WirelessMove(self):
        '''
        up -> forward
        left -> turn_left
        right -> turn_right
        down -> backward
        '''
        if self.key_state[12][1] > 0:
            return 'forward'
        elif self.key_state[15][1] > 0:
            return 'turn_left'
        elif self.key_state[13][1] > 0:
            return 'turn_right'
        elif self.key_state[14][1] > 0:
            return 'backward'
        else:
            return 'stop'

def check_collection(directory, istart=0):
    video_frames = mp42np(os.path.join(directory, "rgb_video.mp4"), way='cv2')
    with open(os.path.join(directory, 'data.json'), 'r') as f:
        data = json.load(f)
        actions = data['true_actions']
    print(f"frames: {video_frames.shape} | actions: {len(actions)}")
    action_space =  ["MoveAhead", "RotateLeft", "RotateRight", "Stop"]
    for i in range(istart, len(video_frames)):
        frame, true_action = video_frames[i], actions[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f'true:{action_space[true_action]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)
        # cv2.putText(frame, f'pred:{action_space[pred_action]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(f'timestep:{i}', frame)
        if cv2.waitKey(1500) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='offline-dataset/maze-dataset')
    parser.add_argument('--is_check', default=False, action='store_true')
    args = parser.parse_args()
    if args.is_check:
        args.record_dir = os.path.join(args.data_dir, f"{len(os.listdir(args.data_dir))-1}")
        check_collection(args.record_dir)
    else:
        args.record_dir = os.path.join(args.data_dir, f"{len(os.listdir(args.data_dir))}")
        args.record_file = os.path.join(args.record_dir, "rgb_video.mp4")
        os.makedirs(args.record_dir, exist_ok=True)
        whlc = WirelessHighLevelController(record_file=args.record_file)
