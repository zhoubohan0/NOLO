import base64
import io
import logging
import os
import re
import time
from collections import deque

import cv2
import numpy as np
import requests
import torch.nn as nn
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

logging.getLogger("httpx").setLevel(logging.WARNING)

# OpenAI API Key
API_KEYs = ['sk-proj-1bsbpd4e3lGE0Z1NdU0VT3BlbkFJ53od3vfGqOAZfaAfxXSB','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']
API_KEY = API_KEYs[1]

# Function to encode the image
def encode_image(image_input):
    """
    Encodes an image from a file path or a NumPy array into a base64 string.

    Args:
        image_input (str or np.ndarray): The image file path or NumPy array.

    Returns:
        str: Base64 encoded string of the image.
    """
    if isinstance(image_input, str) and os.path.exists(image_input):
        with open(image_input, "rb") as image_file:
            img = image_file.read()
    elif isinstance(image_input, np.ndarray):
        buffer = io.BytesIO()
        Image.fromarray(image_input).save(buffer, format='JPEG')
        img = buffer.getvalue()
    else:
        raise ValueError("Input must be a valid file path or a NumPy array.")
    
    return base64.b64encode(img).decode('utf-8')


def image_path_type(image_path):
    return re.match(r"https?://", image_path)


def create_content(image_path_list, prompt):
    image_paths = [p if image_path_type(p) else f"data:image/jpeg;base64,{encode_image(p)}" for p in image_path_list]  # Getting the base64 string

    return [{  # prompt part
            "type": "text",
            "text": prompt
        }] + \
        [{  # image part
        "type": "image_url",
        "image_url": {
            "url": image_path,
            "detail": "low"  # auto, high, low
        }} for image_path in image_paths] 

def select_prominent_frames(video_frames, num_frame):
    """
    Select prominent frames based on the gradients.

    Args:
        video_frames (numpy.ndarray): The video frames of shape (T, H, W, C).
        num_frame (int): Number of frames to keep.

    Returns:
        numpy.ndarray: Video with reduced frames of shape (num_frame, H, W, C).
    """
    T, H, W, C = video_frames.shape
    video_reshape = video_frames.astype(np.float32).reshape(T, -1)
    diff = np.diff(video_reshape, axis=0)
    grad_val = np.linalg.norm(diff, axis=1)
    # var = np.var(diff, axis=1)
    idxs = np.argsort(-grad_val)[:num_frame]  # top-k max
    idxs = np.sort(idxs)
    return video_frames[idxs]

class Pi(nn.Module):
    def __init__(self, dataset, **kwargs):
        super(Pi, self).__init__()
        self.max_tokens = 500
        self.client = OpenAI(api_key=API_KEY)
        # self.pre_prompt = 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: '
        #self.prompt = prompt'Act as a navigation robot to reach the goal frame from the start frame. Just output actions from [forward, left, right] and durations from [1, 2, 3] to reach the goal. The first frame shows the goal, and the second frame shows the start frame. Only output a list of (action, duration) pairs, e.g., [(forward,1), (left,3), (forward,2)]. You are encouraged to imagine more pairs.'
        with open('baseline_policy/gpt4v/prompt.txt') as f:
            self.prompt = f.read().strip()
        # context dataset
        self.context_frames = dataset[0]
        self.representitive_frames = select_prominent_frames(self.context_frames, 2)

    def reset(self,) -> None:
        self.stored_actions = deque()
        self.messages_content = []

    def act(self, observation_dict, timestep):
        if not self.messages_content:  # the first step
            goal_frame_with_text = self.add_text_on_image(observation_dict['goal'], 'Goal')
            self.messages_content = [{"type": "text", "text": self.prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(goal_frame_with_text)}", "detail": "low"}}]

        if not len(self.stored_actions):  # generate a new action
            obs_frame_with_text = self.add_text_on_image(observation_dict['rgb'], 'Start')
            messages=[{
                "role": "user",
                "content": self.append_content(obs_frame_with_text)
            }]
            semantic_action_queue = []
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                result_text = response.choices[0].message.content
                # action, duration = result_text.strip().split(',')
                # print(f"response: {result_text}")
                semantic_action_queue = self.extract_semantic_action_indices(result_text.strip())
                print(f"timestep: {timestep} | semantic_action_queue: {semantic_action_queue}")
            except Exception as e:
                print(e)
            if not semantic_action_queue:
                action = np.random.choice(3)
                duration = np.random.choice(range(1,4), p=[0.1,0.3,0.6])
                semantic_action_queue = [action]*duration
            self.stored_actions.extend(semantic_action_queue)
            # print(f"timestep: {timestep} | semantic_action_queue: {semantic_action_queue}")
            # self.stored_actions['last_action'] = self.extract_action(action.strip())
            # self.stored_actions['duration'] = self.extract_duration(duration.strip())
            # print(f"action: {self.stored_actions['last_action']}, duration: {self.stored_actions['duration']}")
        
        actual_action = self.stored_actions.popleft()
        return actual_action

    def add_text_on_image(self, image, text, way='PIL'):
        if way == 'PIL':
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), text, fill='red',font=ImageFont.truetype("arial.ttf", int(0.15 * image.size[1])))
            # image.save(f'logs/logbaseline/gpt4v/evaluation/{text}.jpg')
            # image.show()
            return np.array(image)
        else:
            # large bold red text on the left top corner, 15% of the image size
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'logs/logbaseline/gpt4v/evaluation/{text}.jpg', image)
            cv2.imshow('image', image)
            cv2.waitKey(5000)
            return image




    def extract_action(self, string):
        string = string.lower()
        one_hot_action = ['forward' in string, 'left' in string, 'right' in string]
        if np.sum(one_hot_action) == 0:
            return np.random.choice(3)
        elif np.sum(one_hot_action) == 1:
            return one_hot_action.index(1)
        else:   
            return np.random.choice(3, p=np.array(one_hot_action,dtype=float)/sum(one_hot_action))
        
    def extract_duration(self, string):
        try:
            return int(string)
        except:
            return np.random.choice(range(1,4), p=[0.1,0.3,0.6])

    def extract_semantic_action_indices(self, string):
        semantic_action_queue = []
        string = string.lower()
        for each in re.findall(r'\(.*?\)', string):
            action, duration = each.strip('()').split(',')
            action, duration = self.extract_action(action), self.extract_duration(duration)
            semantic_action_queue.extend([action]*duration)
        return semantic_action_queue

    def append_content(self, rgb_image):
        return self.messages_content + [{\
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(rgb_image)}",
                    "detail": "low"  
                }}] 


class Pi_with_context(nn.Module):
    def __init__(self, dataset, **kwargs):
        super(Pi_with_context, self).__init__()
        self.max_tokens = 500
        self.client = OpenAI(api_key=API_KEY)
        self.pre_prompt = 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: '
        # self.prompt = 'These are frames about a navigation task. The last frame is the goal, and the second-to-last frame is the current observation. Choose the best action [forward, left, right] and duration [1, 2, 3] to reach the goal based on history frames. Only output a list of (action, duration) pairs, e.g., [(forward,1), (left,3), (forward,2)].'
        with open('baseline_policy/gpt4v/prompt_with_context.txt') as f:
            self.prompt = f.read().strip()
        # context dataset
        self.context_frames = dataset[0]
        self.context_length = 6
        self.representitive_frames = select_prominent_frames(self.context_frames, self.context_length)
        self.history = deque(maxlen=self.context_length+1)

    def encode_image(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    def reset(self,) -> None:
        self.stored_actions = deque()
        self.history.extend([self.encode_image(f) for f in self.representitive_frames])
        self.goal_frame = None

    def act(self, observation_dict, timestep):
        if self.goal_frame is None:  # the first step
            self.goal_frame = [self.encode_image(self.add_text_on_image(observation_dict['goal'], 'Goal'))]
        self.history.append(self.encode_image(observation_dict['rgb']))
        all_frames = list(self.history) + self.goal_frame

        if not len(self.stored_actions):  # generate a new action
            messages=[{
                "role": "user",
                "content": [self.prompt, *map(lambda x: {"image": x, "resize": 224}, all_frames)]
            }]
            semantic_action_queue = []
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                result_text = response.choices[0].message.content
                # action, duration = result_text.strip().split(',')
                # print(f"response: {result_text}")
                semantic_action_queue = self.extract_semantic_action_indices(result_text.strip())
                print(f"timestep: {timestep} | semantic_action_queue: {semantic_action_queue}")
            except Exception as e:
                print(e)
            if not semantic_action_queue:
                action = np.random.choice(3)
                duration = np.random.choice(range(1,4), p=[0.1,0.3,0.6])
                semantic_action_queue = [action]*duration
            self.stored_actions.extend(semantic_action_queue)
            # print(f"timestep: {timestep} | semantic_action_queue: {semantic_action_queue}")
            # self.stored_actions['last_action'] = self.extract_action(action.strip())
            # self.stored_actions['duration'] = self.extract_duration(duration.strip())
            # print(f"action: {self.stored_actions['last_action']}, duration: {self.stored_actions['duration']}")
        
        actual_action = self.stored_actions.popleft()
        return actual_action

    def extract_semantic_action_indices(self, string):
        semantic_action_queue = []
        string = string.lower()
        for each in re.findall(r'\(.*?\)', string):
            action, duration = each.strip('()').split(',')
            action, duration = self.extract_action(action), self.extract_duration(duration)
            semantic_action_queue.extend([action]*duration)
        return semantic_action_queue

    def extract_action(self, string):
        string = string.lower()
        one_hot_action = ['forward' in string, 'left' in string, 'right' in string]
        if np.sum(one_hot_action) == 0:
            return np.random.choice(3)
        elif np.sum(one_hot_action) == 1:
            return one_hot_action.index(1)
        else:   
            return np.random.choice(3, p=np.array(one_hot_action,dtype=float)/sum(one_hot_action))
        
    def extract_duration(self, string):
        try:
            return int(string)
        except:
            return np.random.choice(range(1,4), p=[0.1,0.3,0.6])

    def add_text_on_image(self, image, text, way='PIL'):
        if way == 'PIL':
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), text, fill='red',font=ImageFont.truetype("arial.ttf", int(0.15 * image.size[1])))
            # image.save(f'logs/logbaseline/gpt4v/evaluation/{text}.jpg')
            # image.show()
            return np.array(image)
        else:
            # large bold red text on the left top corner, 15% of the image size
            cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'logs/logbaseline/gpt4v/evaluation/{text}.jpg', image)
            cv2.imshow('image', image)
            cv2.waitKey(5000)
            return image


if __name__ == '__main__':
    max_tokens = 30
    image_path_list = ['/home/zbh/Desktop/zbh/Proj/nav/my/oso/offline-dataset/robothor-dataset/900/val/FloorPlan_Train1_5/goal_frames/FloorPlan_Train1_5_AlarmClock_109.png',
                       '/home/zbh/Desktop/zbh/Proj/nav/my/oso/offline-dataset/robothor-dataset/900/val/FloorPlan_Train1_5/goal_frames/FloorPlan_Train1_5_AlarmClock_110.png']
    pre_prompt = 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: '
    prompt = pre_prompt+'The goal of the navigation task is depicted by the first frame. The second frame is currently observed. Select the most appropriate action from the options [forward, move left, move right] that will help reach the goal. Provide the selected action directly.'
    # payload = {
    #     "model": "gpt-4-vision-preview",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": create_content(image_path_list, prompt)
    #         }
    #     ],
    #     "max_tokens": max_tokens
    # }
    # b = time.time()
    # try:
    #     response = requests.post("https://api.openai.com/v1/engines/gpt-4.0/completions", headers={"Content-Type": "application/json","Authorization": f"Bearer {API_KEY}"}, json=payload)
    #     print(response.json())
    # except Exception as e:
    #     print(e)
    # print(f'Time: {time.time() - b} seconds')
    client = OpenAI(api_key=API_KEY)
    print("-----------------------------------begin------------------------------------------")
    messages=[{
        "role": "user",
        "content": create_content(image_path_list, prompt),
    }]
    begin = time.time()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=max_tokens,
    )
    result_text = response.choices[0].message.content
    print(f"response: {result_text}")
    print(time.time() - begin)
    print("------------------------------------end-------------------------------------------")
    