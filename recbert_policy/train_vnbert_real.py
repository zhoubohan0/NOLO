import argparse, cv2
import os.path as osp
import random
from datetime import timedelta
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from tensorboardX import SummaryWriter
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import ColorJitter, Compose, RandomResizedCrop, ToPILImage, Resize

from recbert_policy.vnbert import VNBERTPolicy
from utils.basic_utils import (
    mp42np,
    np2mp4,
    update_args_from_json,
    read_json,
    save_for_visualize,
    save_json,
    seed_everything,
    remove_similar_frames,
    select_prominent_frames
)

def load_SA(data_dir):
    all_data = read_json(osp.join(data_dir,'data.json'))
    action_indices = all_data['true_actions'][1:]  # TODO 
    video_frames = mp42np(osp.join(data_dir, 'rgb_video.mp4'), way='cv2')
    video_frames = video_frames[:-(len(video_frames)-len(action_indices))]  # drop the last frame corresponding to the "STOP"
    return video_frames, action_indices 

class NewDataset(Dataset):
    def __init__(self, data_dir, num_action=9, horizon=30):
        self.data_dir = data_dir
        self.num_action = num_action
        self.horizon = horizon
        self.augment = Compose([
            ToPILImage(),
            RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        self.switch_dataset()

    def check_video_action(self, video_frames, actions, istart=0):
        print(f"frames: {len(video_frames)} | actions: {len(actions)}")
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

    def load_dataset(self, data_dir):
        self.video_frames, self.action_indices = load_SA(data_dir)
        # self.check_video_action(self.video_frames, self.action_indices, 0)
        if self.num_action > 3: self.action_indices = self.action_with_duration(self.action_indices)
        self.action_indices = torch.LongTensor(self.action_indices)
        # sample context
        self.context_ids = torch.arange(len(self.video_frames))
        self.context_frames = torch.from_numpy(np.stack([np.array(self.augment(self.video_frames[i])) for i in self.context_ids]))
        self.context_actions = self.action_indices[self.context_ids]

    def switch_dataset(self,):
        cur_data_dir = random.choice(glob(osp.join(self.data_dir, '*')))
        if not hasattr(self, 'cur_data_dir') or cur_data_dir != self.cur_data_dir:
            self.cur_data_dir = cur_data_dir
            self.load_dataset(self.cur_data_dir)

    def action_with_duration(self, actions, k=3):
        new_actions = []
        cur_action, duration = actions[0], 1
        for i in range(1,len(actions)):
            if actions[i] == cur_action:
                duration += 1
            else:  
                new_actions.extend([cur_action * k + min(j,k)-1 for j in range(duration,0,-1)])
                cur_action, duration = actions[i], 1
        new_actions.extend([cur_action * k + min(j,k)-1 for j in range(duration,0,-1)])
        assert len(new_actions) == len(actions)
        return new_actions

    def __getitem__(self, idx):
        if idx < self.horizon:
            idx = np.random.choice(range(self.horizon, len(self.action_indices)))
        goal = torch.from_numpy(np.array(self.augment(self.video_frames[idx])))
        idxs = np.arange(idx-self.horizon, idx)
        state = torch.from_numpy(np.stack([np.array(self.augment(self.video_frames[i])) for i in idxs]))
        action = self.action_indices[idxs]
        return goal, state, action

    def __len__(self):
        return len(self.action_indices)
    

class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # dataloader
        self.dataset = NewDataset(args.data_dir, num_action=args.num_action, horizon=args.horizon)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

        # model
        self.policy = VNBERTPolicy(args.num_action, args.action_emb_size, args.temporal_net).to(self.device)

        # optimizer
        self.optimizer = self.configure_optimizers()

        # logging
        self.save_dir = f'{args.save_dir}/{args.exp_name}'
        self.writer = SummaryWriter(self.save_dir)
        train_config_dict = vars(args)
        save_json(train_config_dict, f'{self.save_dir}/config.json')
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in train_config_dict.items()]),),
        )

        # load pretrain ckpt
        self.load(args.pretrain_ckpt)

        # half precision
        self.scaler = GradScaler() if args.half_precision else None

    def configure_optimizers(self,):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.policy.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.policy.named_parameters()}
        inter_params = decay & no_decay
        # union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.args.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.lr, betas=self.args.betas)
        return optimizer

    def load(self, pretrain_ckpt=''):
        if pretrain_ckpt:  # soft load, only load the common keys
            ckpt = torch.load(pretrain_ckpt)
            state_dict = self.policy.state_dict()
            filtered_ckpt = {k: v for k, v in ckpt.items() if k in state_dict and state_dict[k].shape == v.shape}
            is_partial = len(filtered_ckpt) < len(ckpt) 
            state_dict.update(filtered_ckpt)
            self.policy.load_state_dict(state_dict,strict=True)
            print(f'{"[ Partially ]" if is_partial else ""} load pretrain ckpt from {pretrain_ckpt}') 

    def roll(self, action, k):
        return action // 3 * 3 + torch.maximum(action % 3 - k, torch.zeros_like(action))

    def cal_loss(self, goal, state, action):
        loss_dict = {}
        B, T, A = state.size(0), state.size(1), self.args.num_action

        q_logits, a_logits = torch.empty((B,T,self.policy.hidden_size), device=self.device), torch.empty((B,T,A), device=self.device)
        st, enc_context, context_frame_embs = self.policy.enc_context(self.dataset.context_frames.to(self.device), self.dataset.context_actions.to(self.device))  # (1, 1, Dc), (1, Tc+1, Dc), (Tc, Dcf)
        enc_pure_context = enc_context[:,1:,:]  # (1, Tc, Dc)
        goal_repr = self.policy.goal_encoder(goal) 

        for t in range(self.args.horizon):
            enc_context = torch.cat((st, enc_pure_context.repeat(len(st),1,1)), dim=1)
            st, logit = self.policy.enc_step(enc_context, goal_repr, state[:,t])  # (B, Dc), (B, A)
            q_logits[:,t:t+1], a_logits[:,t] = st, logit
        
        # loss action
        '''
        loss_dict['loss_action'] = F.cross_entropy(a_logits.view(-1,A), action.view(-1))
        '''
        loss_dict['loss_action'] = 0.6 * F.cross_entropy(a_logits.view(-1,A), action.view(-1)) + \
            0.3 * F.cross_entropy(a_logits.view(-1,A), self.roll(action,1).view(-1)) + \
            0.1 * F.cross_entropy(a_logits.view(-1,A), self.roll(action,2).view(-1))

        # loss termination
        termination = torch.zeros((B, T, 1),dtype=torch.long).to(self.device).detach()
        termination[:,-1] = 1
        terminal_logits = self.policy.termination_head(q_logits)
        loss_dict['loss_termination'] = F.cross_entropy(terminal_logits.view(-1,2), termination.view(-1))

        # loss temporal
        if self.policy.temporal_predictor is not None:
            t_idxs = self.dataset.context_ids.to(self.device)
            permt_idxs = t_idxs[torch.randperm(len(self.dataset.context_ids))].to(self.device)
            loss_dict['loss_temporal'] = 0.5 * F.binary_cross_entropy_with_logits(
                torch.cat([self.policy.temporal_predictor(context_frame_embs), self.policy.temporal_predictor(context_frame_embs[permt_idxs])], dim=-1), 
                F.one_hot((t_idxs < permt_idxs).long()).float()
            )

        # loss bcq
        '''
        q_value = self.policy.critic_head(q_logits)  # (B, T, A)
        cur_Q = q_value.gather(-1, action.unsqueeze(-1))
        with torch.no_grad():
            # next
            logits_q_next = q_value.roll(-1, 1)
            probs_a_next = F.log_softmax(a_logits,-1).exp().roll(-1, 1)
            probs_a_next = (probs_a_next / probs_a_next.max(-1, keepdim=True)[0] > self.args.action_thresh).float()
            next_actions = (probs_a_next * logits_q_next + (1 - probs_a_next) * -1e8).argmax(-1, keepdim=True)
            # TD_target = rewards + gamma * nextqtarget(nexts, nexta)
            logits_q_target_next = self.policy.target_critic_head(q_logits.roll(-1, 1))
            TD_target = termination + self.args.gamma * logits_q_target_next.gather(-1, next_actions) * (1 - termination)
        loss_dict['loss_bcq'] = F.smooth_l1_loss(cur_Q, TD_target)   
        loss_dict['loss_action_logits_norm'] = 0.01 * a_logits.pow(2).mean()
        '''
        return loss_dict

    def rollout(self):
        seed_everything(42)
        num_batch_update = 0
        for epoch in range(1,1+self.args.epoch):
            for goal, state, action in self.dataloader:
                goal, state, action = goal.to(self.device), state.to(self.device), action.to(self.device) # (B, H, W, C), (B, T, H, W, C), (B,T)

                if self.scaler is not None:
                    with autocast(dtype=torch.float16):
                        loss_dict = self.cal_loss(goal, state, action)
                    loss = sum([1.0 * v for k, v in loss_dict.items()])
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()  
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict = self.cal_loss(goal, state, action)
                    loss = sum([1.0 * v for k, v in loss_dict.items()])
                    self.optimizer.zero_grad()
                    loss.backward()  
                    self.optimizer.step()

                self.policy.polyak_target_update()

                # save log
                num_batch_update += 1
                for k,v in loss_dict.items():
                    self.writer.add_scalar(f'loss/{k}', v.item(), num_batch_update)
                
                if num_batch_update % self.args.log_interval == 0:
                    scene_name = osp.basename(self.dataset.cur_data_dir)
                    print(f'Epoch:{epoch} | Iteration: {num_batch_update:05d} | Scene: {scene_name}\t|\t{" | ".join([f"{k}: {v.item():.6f}" for k,v in loss_dict.items()])}')
                
                if num_batch_update % self.args.switch_interval == 0:
                    self.dataset.switch_dataset()
                
                if num_batch_update % self.args.save_interval == 0:
                    torch.save(self.policy.state_dict(), osp.join(self.save_dir,f'policy_{num_batch_update}.pth'))
                    print(f'Iteration: {num_batch_update:05d} | Model saved')
                
        self.writer.close()
        

class Visualizer(nn.Module):
    def __init__(self,ckpt_file,args,):
        super().__init__()
        assert ckpt_file, 'Please specify the pretrain_ckpt'
        args = update_args_from_json(args, osp.join(osp.dirname(args.pretrain_ckpt), 'config.json'))
        # args.data_dir = f'offline-dataset/mp3d-dataset/900/val'
        self.policy = VNBERTPolicy(args.num_action, args.action_emb_size, args.temporal_net).to(args.device)
        self.dataset = NewDataset(args.data_dir,num_action=args.num_action, horizon=args.horizon)

        self.load(ckpt_file)

    def load(self, pretrain_ckpt=''):
        if pretrain_ckpt:
            ckpt = torch.load(pretrain_ckpt, map_location='cpu')
            self.policy.load_state_dict(ckpt,strict=True)
            print(f'Load checkpoint from {pretrain_ckpt}')

    def visualize_context(self, scene_dir, num_sample=300):
        self.dataset.load_dataset(scene_dir)
        cam_image_list = np.empty((900, 224, 224, 3), dtype=np.uint8)
        timesteps = np.arange(0,900)
        for i in range(0,900,num_sample):
            cam_timesteps = np.arange(i, i+num_sample)
            cam_images = self.policy.visualize_emb(torch.from_numpy(self.dataset.video_frames[cam_timesteps]).cuda())
            cam_image_list[cam_timesteps] = cam_images
        save_for_visualize(
            cam_image_list, 
            ['']*900,#timesteps, 
            save_file=osp.join('visualization/goal-context-attention', f'{osp.basename(scene_dir)}.mp4')
        )
        import pdb; pdb.set_trace()
        print(f'Visualize embedding for {scene_dir}')

    def visualize_embedding(self, scene_dir, num_sample=900, tag=''):
        self.dataset.load_dataset(scene_dir)
        # TDOO: 1. state embeddng sequence 2. initial context embedding sequence
        # X = torch.from_numpy(self.dataset.video_frames)  
        X = self.dataset.context_frames
        # context_enc0, context_emb = self.policy.enc_context(X.cuda())
        # vis_target = context_enc0

        vis_target = np.empty((900, 512))
        for i in range(0,900,num_sample):
            with torch.no_grad():
                x = X[i:i+num_sample].cuda()
                context_emb = self.policy.state_encoder(x)
                # context_enc0, context_emb = self.policy.enc_context(x)
            vis_target[i:i+num_sample] = context_emb.cpu().detach().numpy()
        
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        orange, cyan = np.array([245,150,55])/255, np.array([135,200,195])/255
        params = {
            "font.size": 20,
            'font.family': 'STIXGeneral',
            "figure.subplot.wspace": 0.2,
            "figure.subplot.hspace": 0.4,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "legend.fontsize": 17,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.direction": 'out',
            "ytick.direction": 'out',
        }
        plt.rcParams.update(params)
        dr = TSNE(n_components=2, init='pca', random_state=0, perplexity=30.0, n_iter=1000, verbose=0)
        # dr = PCA(n_components=2)
        vis_embs = dr.fit_transform(vis_target)
        # vis_embs = vis_embs[::2]  # downsample
        plt.figure(figsize=(8, 6))
        plt.scatter(vis_embs[:, 0], vis_embs[:, 1], marker='o', alpha=0.99, s=20,color=cyan)#, cmap='viridis', c='b'
        # save_file = osp.join('visualization/embedding-sequence', tag,f'{osp.basename(scene_dir)}_{tag}.png')
        # if not osp.exists(osp.dirname(save_file)):
        #     os.makedirs(osp.dirname(save_file))
        # plt.savefig(save_file, dpi=300)
        plt.savefig('visualization/embedding-sequence/robothor_nocontext/Val1_1_nocontext.png', dpi=600)
        print(f'Successfully visualize embedding for {scene_dir}')
        plt.close()
        # plt.show()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='robothor')
    parser.add_argument('--data_dir', type=str, default="offline-dataset/maze-dataset")
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--pretrain_ckpt', '-c', type=str, default='')
    parser.add_argument('--num_action', type=int, default=9)
    parser.add_argument('--action_emb_size', type=int, default=256)
    parser.add_argument('--action_thresh', type=float, default=0.5)
    parser.add_argument('--temporal_net', type=str, default='none')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--betas', type=float, default=(0.9, 0.95))
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=2000000)
    parser.add_argument('--batch_size', type=int, default=6)  # as large as possible
    parser.add_argument('--horizon', type=int, default=30)      # as small as possible
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--switch_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--half_precision', type=int, default=0)
    
    args, unknown = parser.parse_known_args()
    args.device = f'cuda'

    mode = 'train'#'vis'
    if mode == 'train':
        # dist.init_process_group(backend='nccl', timeout=timedelta(seconds=60),)
        # local_rank = dist.get_rank()
        # n_gpu = dist.get_world_size()
        # args.device = f'cuda:{local_rank}'
        trainer = Trainer(args)
        trainer.rollout()

    elif 'vis' in mode:
        # TODO: 1. trained room  2. trained room unseen layout  3. unseen room
        visualizer = Visualizer(args.pretrain_ckpt,args=args)
        # val_dir = 'offline-dataset/robothor-dataset/900/val/FloorPlan_Train1_5'
        val_dir = 'offline-dataset/robothor-dataset/900/val/FloorPlan_Val1_1'
        # val_dir = 'offline-dataset/mp3d-dataset/900/val/gTV8FGcVJC9'
        visualizer.visualize_context(val_dir)

        # visualizer.visualize_embedding(val_dir, tag=args.domain+''if 'ele' in args.pretrain_ckpt else args.domain+'_nocontext')

        # for val_dir in glob(osp.join('offline-dataset/robothor-dataset/900/val/*')):
            # visualizer.visualize_embedding(val_dir, tag=args.domain+''if 'ele' in args.pretrain_ckpt else args.domain+'_nocontext')
