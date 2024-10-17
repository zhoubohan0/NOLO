import random, torch, numpy as np, imageio, json, cv2, os#, decord

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def mp42np(mp4_file, way='ffmpeg'):
    if way == 'decord':
        return np.stack([im.asnumpy() for im in decord.VideoReader(mp4_file, num_threads=16)])
    elif way == 'ffmpeg':
        return np.stack([im for im in imageio.get_reader(mp4_file,  'ffmpeg')])
    elif way == 'cv2':
        cap = cv2.VideoCapture(mp4_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def numpy2tensor(x):
    return torch.from_numpy(np.stack(x))

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


def save_for_visualize(frame_images, frame_idxs, save_file):
    for i, cam_image in enumerate(frame_images):
        cv2.putText(cam_image, f'{frame_idxs[i]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    np2mp4(frame_images, save_file, fps=2)

def update_args_from_json(args, json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        setattr(args, key, value)
    return args

def remove_similar_frames(video_frames, num_frame, percent=90):
    """
    Remove similar adjacent frames based on cosine similarity.

    Args:
        video_frames (numpy.ndarray): The video frames of shape (T, H, W, C).
        num_frame (int): Number of frames to keep.
        round (int): Number of rounds to remove similar frames.

    Returns:
        numpy.ndarray: Video with reduced frames of shape (num_frame, H, W, C).
    """
    T, H, W, C = video_frames.shape
    rtn_indices = np.arange(T)
    if T <= num_frame: return rtn_indices
    video_frames = video_frames.copy().reshape(T, -1).astype(np.float32)
    video_frames /= np.linalg.norm(video_frames, axis=1, keepdims=True)
    for _ in range(T - num_frame):
        cos_sim = np.einsum('ij,ij->i', video_frames[rtn_indices[:-1]], video_frames[rtn_indices[1:]])
        percentile = np.percentile(cos_sim, percent)
        mask = list(cos_sim < percentile) + [True]
        while mask.count(True) < num_frame:
            percent += 1
            mask = list(cos_sim < np.percentile(cos_sim, percent)) + [True]
        rtn_indices = rtn_indices[mask]
        if len(rtn_indices) == num_frame:
            break
        # print(f'Round: {_} | Frames: {len(rtn_indices)} | Percentile: {percentile:.6f} | Percent: {percent:.6f}')
    return rtn_indices


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
    video_frames = video_frames.astype(np.float32).reshape(T, -1)
    diff = np.diff(video_frames, axis=0)
    grad_val = np.linalg.norm(diff, axis=1)
    # var = np.var(diff, axis=1)
    return np.argsort(-grad_val)[:num_frame]  # top-k max
