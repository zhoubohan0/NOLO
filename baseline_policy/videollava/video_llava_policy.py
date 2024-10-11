import base64
import dataclasses
import json
import os.path as osp
import re
from collections import deque
from enum import Enum, auto
from io import BytesIO
from typing import List, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import StoppingCriteria  # pipeline

# from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from videollava.conversation import conv_templates, SeparatorStyle
# from videollava.utils import disable_torch_init
from .constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .model.builder import load_pretrained_model


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)









def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def mp42np(mp4_file, way='ffmpeg'):
    if way == 'decord':
        return np.stack([im.asnumpy() for im in decord.VideoReader(mp4_file, num_threads=16)])
    elif way == 'ffmpeg':
        return np.stack([im for im in imageio.get_reader(mp4_file,  'ffmpeg')])

def get_model_and_processor(model_path='LanguageBind/Video-LLaVA-7B'):
    # disable_torch_init()
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=True, device='cuda')
    video_processor = processor['video']
    conv = conv_llava_v1.copy()#conv_templates["llava_v1"].copy()
    return model, tokenizer, video_processor, conv

def extract_action(string):
    string = string.lower()
    one_hot_action = ['forward' in string, 'left' in string, 'right' in string]
    if np.sum(one_hot_action) == 0:
        return np.random.choice(3)
    elif np.sum(one_hot_action) == 1:
        return one_hot_action.index(1)
    else:   
        return np.random.choice(3, p=np.array(one_hot_action,dtype=float)/sum(one_hot_action))

class Pi(nn.Module):
    def __init__(self, dataset,**kwargs):
        super(Pi, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # context dataset
        self.context_frames = dataset[0]
        # load model 
        self.model, self.tokenizer, self.video_processor, conv = get_model_and_processor()
        
        # process prompt
        prompt = 'The goal of the video is represented by the first frame. The subsequent frames show the history. Based on the last frame of the history, select the most appropriate action from the options [forward, move left, move right] that will help reach the goal depicted in the first frame of the video. Provide the selected action directly.'
        # prompt = 'Given a navigation video where the first frame shows the goal and subsequent frames depict an agent executing actions [forward, move left, move right], determine the most efficient action to take based on the current observation (last frame) and the history. Additionally, specify the number of steps required for the selected action to reach the goal.'
        prompt = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + prompt
        conv.append_message(conv.roles[0], prompt), conv.append_message(conv.roles[1], None)
        self.input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        self.stopping_criteria = KeywordsStoppingCriteria([conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2], self.tokenizer, self.input_ids)

    def reset(self, capacity=8):
        self.obs_deque = deque(self.context_frames[-capacity:],maxlen=capacity)
        self.stored_actions = dict(last_action=-1,duration=-1,action_prob=0)

    def act(self, observation_dict, timestep):
        self.obs_deque.append(observation_dict['rgb'])
        if self.stored_actions['duration'] == -1:  # generate a new action
            video_tensor = self.video_processor(np.stack([observation_dict['goal']]+list(self.obs_deque)[1:]), return_tensors='pt')['pixel_values']
            with torch.inference_mode():
                output_ids = self.model.generate(self.input_ids, images=video_tensor.to(self.device, dtype=torch.float16), do_sample=True, temperature=0.9, max_new_tokens=1024, use_cache=True, stopping_criteria=[self.stopping_criteria])
            outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()
            action_index = extract_action(outputs)
            self.stored_actions['last_action'] = action_index 
            self.stored_actions['duration'] = np.random.choice(3, p=[0.6,0.3,0.1])

        actual_action = self.stored_actions['last_action']
        self.stored_actions['duration'] -= 1
        return actual_action

def videoQA(video_file, prompt):
    # load model    
    modelgroup = get_model_and_processor()
    model, tokenizer, video_processor, conv = modelgroup
    # calculate the parameters of the model
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # process prompt
    prompt = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + prompt
    conv.append_message(conv.roles[0], prompt), conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stopping_criteria = KeywordsStoppingCriteria([conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2], tokenizer, input_ids)
    # process video
    actions = read_json(osp.join(osp.dirname(video_file),'data.json'))['action_indices']
    frames = mp42np(video_file)
    for temp in np.arange(0.1, 1.1, 0.1):
        acc, num = 0, 50
        for b in np.random.choice(len(frames)-8,num):
            video_tensor = video_processor((frames[:], frames[b: b+7], frames[b+7:b+8]), return_tensors='pt')['pixel_values']
            # tensor = [v.to(model.device, dtype=torch.float16) for v in video_tensor] if isinstance(video_tensor, list) else video_tensor.to(model.device, dtype=torch.float16)  # (B=1, C, T, H, W)
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=video_tensor.to(model.device, dtype=torch.float16), do_sample=True, temperature=temp, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            action_index = extract_action(outputs)
            # print(outputs,'\t',action_index,'\t',actions[b+6])
            acc += action_index == actions[b+6]
        print('temp:',temp,'acc:',acc/num)
    return outputs


if __name__ == '__main__':
    video_file = '/home/zbh/Desktop/zbh/Proj/nav/my/oso/offline-dataset/robothor-dataset/900/val/FloorPlan_Train{i}_5/rgb_video.mp4'
    prompt = 'Given a navigation video where the final frame shows the goal and prior frames show the history, select the best action [forward, move left, move right] and step duration [1, 2, 3] based on the current observation (last frame of the history) to reach the goal efficiently. Output the action and duration directly.'
    for i in range(1, 13):
        video_file = video_file.format(i=i)
        answer = videoQA(video_file, prompt)