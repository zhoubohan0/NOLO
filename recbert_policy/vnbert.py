from __future__ import absolute_import, division, print_function, unicode_literals

import math
import pdb
import sys
from collections import deque
from copy import deepcopy

import clip
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision import models
from torchvision.transforms.v2 import Compose, Normalize
from transformers import BertConfig, BertPreTrainedModel

from utils.basic_utils import remove_similar_frames
# from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM, GradCAM, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, HiResCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm  # TODO
# except (ImportError, AttributeError) as e:
#     logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
# BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = True

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        return self.activation(self.dense(first_token_tensor))


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input):
        ''' Cross Attention -- cross for vision not for language '''
        return self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)

    def self_att(self, visn_input, visn_attention_mask):
        ''' Self Attention -- on visual features with language clues '''
        return self.visn_self_att(visn_input, visn_attention_mask)

    def output_fc(self, visn_input):
        ''' Feed forward '''
        return self.visn_output(self.visn_inter(visn_input), visn_input)

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        # (B, Tc, Dc), (B, 1, 1, Tc), (B, Tv, Dv'), (B, 1, 1, Tv)
        
        ''' visual self-attention with state '''
        visn_att_output = torch.cat((lang_feats[:, 0:1, :], visn_feats), dim=1)                     # (B, 1+Tv, D)
        state_vis_mask = torch.cat((lang_attention_mask[:,:,:,0:1], visn_attention_mask), dim=-1)   # (B, 1, 1, 1+Tv)

        ''' state and vision attend to language '''
        # (B, 1+Tv, D), (B, num_head, 1+Tv, Tc-1)
        visn_att_output, cross_attention_scores = self.cross_att(lang_feats[:, 1:, :], lang_attention_mask[:, :, :, 1:], visn_att_output)  
        
        language_attention_scores = cross_attention_scores[:, :, 0, :]  # (B, num_head, Tc-1)

        state_visn_att_output, self_attn_scores = self.self_att(visn_att_output, state_vis_mask) # (B, 1+Tv, D), (B, num_head, 1+Tv, 1+Tv)
        state_visn_output = self.output_fc(state_visn_att_output)                                # (B, 1+Tv, D)

        visn_att_output = state_visn_output[:, 1:, :]                                            # (B, Tv, D)
        lang_att_output = torch.cat((state_visn_output[:, 0:1, :], lang_feats[:,1:,:]), dim=1)   # (B, Tc, D), cat new state with old context

        visual_attention_scores = self_attn_scores[:, :, 0, 1:]                                  # (B, num_head, Tv)

        return lang_att_output, visn_att_output, language_attention_scores, visual_attention_scores  # (B, Tc, D), (B, Tv, D), (B, num_head, Tc-1), (B, num_head, Tv)


class VisionEncoder(nn.Module):
    def __init__(self, vision_size, config):
        super().__init__()
        feat_dim = vision_size

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)

        output = self.dropout(x)
        return output


class VNBertModule(BertPreTrainedModel):
    def __init__(self, config):
        super(VNBertModule, self).__init__(config)

        self.img_feature_type = config.img_feature_type  # ''
        self.vl_layers = config.vl_layers                # 4
        self.la_layers = config.la_layers                # 9
        self.lalayer = nn.ModuleList([BertLayer(config) for _ in range(self.la_layers)])
        self.addlayer = nn.ModuleList([LXRTXLayer(config) for _ in range(self.vl_layers)])
        self.vision_encoder = VisionEncoder(config.input_dim, config)
        self.pooler = BertPooler(config)
        # self.apply(self.init_weights)

    def forward(self, mode, context_emb, gsa_emb=None):
        '''
        context_emb: (B, Tc, Dc)
        gsa_emb: (B, N, Dg+Ds+Da)
        '''
        # context_mask = -10000.0 * (1.0 - rearrange(lang_mask, 'b t -> b 1 1 t').to(dtype=next(self.parameters()).dtype))  # (B, 1, 1, T), fp16 compatibility
        context_mask = torch.zeros(context_emb.shape[0], 1, 1, context_emb.shape[1]).to(dtype=next(self.parameters()).dtype).to(context_emb.device)  # (B, 1, 1, Tc)
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            text_embeds = context_emb  # (B, Tc, Dc)

            for layer_module in self.lalayer:
                text_embeds, attention_output = layer_module(text_embeds, context_mask)  # (B, Tc, Dc)

            return self.pooler(text_embeds), text_embeds  # (B, Dc), (B, Tc, Dc), "pooler" takes the hidden state corresponding to the first token

        elif mode == 'visual':
            ''' LXMERT visual branch (no language processing during navigation) '''
            B = gsa_emb.shape[0]
            if len(context_emb) != B:
                context_emb = context_emb.repeat(B, 1, 1)
                context_mask = context_mask.repeat(B, 1, 1, 1)
            visn_output = self.vision_encoder(gsa_emb)  # (B, Tv, Dv) -> (B, Tv, Dv')
            # vis_mask = -10000.0 * (1.0 - rearrange(vis_mask, 'b t -> b 1 1 t').to(dtype=next(self.parameters()).dtype))  # (B, 1, 1, Tv)
            vis_mask = torch.zeros(B, 1, 1, gsa_emb.shape[1]).to(dtype=next(self.parameters()).dtype).to(gsa_emb.device)  # (B, 1, 1, Tv)
            for layer_module in self.addlayer:
                # (B, Tc, Dc), (B, Tv, Dv'), (B, num_head, Tc-1), (B, num_head, Tv)
                context_emb, visn_output, language_attention_scores, visual_attention_scores = layer_module(context_emb, context_mask, visn_output, vis_mask)

            language_state_scores = language_attention_scores.mean(dim=1)  # (B, Tc)
            visual_action_scores = visual_attention_scores.mean(dim=1)     # (B, Tv)

            # weighted_feat
            language_attention_probs = nn.Softmax(dim=-1)(language_state_scores.clone()).unsqueeze(-1)  # (B, Tc-1, 1)
            visual_attention_probs = nn.Softmax(dim=-1)(visual_action_scores.clone()).unsqueeze(-1)     # (B, Tv, 1)

            attended_language = (language_attention_probs * context_emb[:, 1:, :]).sum(1)  # (B, Tc-1, 1), (B, Tc-1, Dc) -> (B, Dc), the first token changes and other context tokens remain  
            attended_visual = (visual_attention_probs * visn_output).sum(1)                   # (B, Tv, 1), (B, Tv, Dv') -> (B, Dv')  

            return self.pooler(context_emb), visual_action_scores, attended_language, attended_visual   # (B, Dc), (B, Tv), (B, Dc), (B, Dv')
        


def get_vlnbert_modules(input_dim):
    vis_config = BertConfig.from_pretrained('bert-base-uncased')#, local_files_only=True
    # TODO: modify the original bert config
    vis_config.input_dim = input_dim
    vis_config.img_feature_type = ""
    vis_config.vl_layers = 4
    vis_config.la_layers = 9
    visual_model = VNBertModule.from_pretrained('Prevalent/pretrained_model/pytorch_model.bin', config=vis_config, local_files_only=True, ignore_mismatched_sizes=True)
    return visual_model

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Place365ResEncoder(nn.Module):
    def __init__(self, arch='resnet18', shape=(224,224), mu_sigma=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], pretrained=True):
        super(Place365ResEncoder, self).__init__()
        resnet = models.__dict__[arch](num_classes=365)#weights=False,
        if pretrained:
            checkpoint = torch.load(f'recbert_policy/{arch}_places365.pth.tar', map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
        if arch == 'resnet50':
            self.linear = nn.Linear(2048, 512)
            nn.init.constant_(self.linear.weight, 0.005)
        self.net = nn.Sequential(*list(resnet.children())[:-1])
        self.obs_transform = Compose([
            # Resize(shape),# interpolation=InterpolationMode.NEAREST_EXACT,
            Normalize(mu_sigma[0], mu_sigma[1], inplace=True),
        ])
        # self.net.eval()

    def forward(self, x):
        '''(B, H, W, C) -> (B, emb_dim)'''
        x = x.permute(0,3,1,2) / 255.
        x = self.obs_transform(x)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        if hasattr(self, 'linear'):
            x = self.linear(x)
        return x
    

class ClipEncoder(nn.Module):
    def __init__(self, shape=(224,224), mu_sigma=[(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)]):
        super(ClipEncoder, self).__init__()
        self.clip_model_encoder, self.clip_preprocessor = clip.load("RN50")  # ViT-B/16, 32 is larger than 16
        self.clip_preprocessor = Compose([
            # Resize(shape, interpolation=PIL.Image.BICUBIC),
            Normalize(mu_sigma[0], mu_sigma[1], inplace=True),
        ])
        self.clip_model_encoder.eval()  # fix clip parameters
        
    def forward(self, x):
        '''(B, H, W, C) -> (B, emb_dim)'''
        x = x.permute(0,3,1,2) / 255.
        x = self.clip_preprocessor(x)
        with torch.no_grad():
            if x.device.type == 'cpu':
                x = self.clip_model_encoder.to(torch.float32).encode_image(x)
            else:  # for cuda
                x = self.clip_model_encoder.encode_image(x)  
        return x


class VNBERTPolicy(nn.Module):
    def __init__(self, num_action=9, action_emb_size=256, temporal_net='none'):
        super(VNBERTPolicy, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')
        obs_emb_dim = 512
        goal_emb_dim = 1024

        self.vln_bert = get_vlnbert_modules(obs_emb_dim + goal_emb_dim + action_emb_size)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        self.hidden_size = self.vln_bert.config.hidden_size  # 768
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        # self.angle_feat_size = angle_feat_size
        # self.action_state_project = nn.Sequential(nn.Linear(hidden_size+angle_feat_size, hidden_size), nn.Tanh())
        # self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        # self.drop_env = nn.Dropout(p=0.3)
        # self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        # self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(self.hidden_size, eps=layer_norm_eps)

        
        # Encoder for state and goal
        self.num_action = num_action
        self.state_encoder = Place365ResEncoder(pretrained=False)
        self.goal_encoder = ClipEncoder()

        # action embedding 
        self.action_emb_layer = nn.Sequential(nn.Embedding(num_action, action_emb_size), nn.Tanh())
        nn.init.normal_(self.action_emb_layer[0].weight, mean=0.0, std=0.02)
        
        # temporal predictor
        if temporal_net == 'ranknet':
            self.temporal_predictor = nn.Sequential(
                spectral_norm(nn.Linear(obs_emb_dim, obs_emb_dim // 4)),
                nn.SiLU(),
                spectral_norm(nn.Linear(obs_emb_dim // 4, obs_emb_dim // 16)),
                nn.SiLU(),
                nn.Linear(obs_emb_dim // 16, 1)
            )
        elif temporal_net == 'ele':
            self.temporal_predictor = nn.Sequential(
                spectral_norm(nn.Linear(2 * obs_emb_dim, obs_emb_dim // 2)),
                nn.SiLU(),
                spectral_norm(nn.Linear(obs_emb_dim // 2, obs_emb_dim // 8)),
                nn.SiLU(),
                spectral_norm(nn.Linear(obs_emb_dim // 8, obs_emb_dim // 32)),
                nn.SiLU(),
                nn.Linear(obs_emb_dim // 32, 1),
            )
        elif temporal_net == 'none':
            self.temporal_predictor = None
        else:
            raise NotImplementedError

        # critic head
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_action),
        )
        self.target_critic_head = deepcopy(self.critic_head)  

        # termination head
        self.termination_head = nn.Linear(self.hidden_size, 2, bias=False)

    def enc_context(self, context_frames, context_actions=None):
        '''
        context_frames: (Tc, H, W, C)
        context_actions: (Tc)
        '''
        context_frame_embs = self.state_encoder(context_frames)                             # (Tc, Dcf)
        if context_actions is not None:
            context_action_embs = self.action_emb_layer(context_actions).unsqueeze(0)        # (1, Tc, Dca)
            context_embs = torch.cat((context_frame_embs.unsqueeze(0), context_action_embs), dim=-1)      # (1, Tc, Dcf+Dca)
        else:
            context_embs = context_frame_embs.unsqueeze(0)                                                # (1, Tc, Dc)
        first_special_token = torch.zeros(1, 1, context_embs.shape[-1]).to(context_embs.device)  # (1, 1, Dc)
        context_embs = torch.cat([first_special_token, context_embs], dim=1)                   # (1, Tc+1, Dc)
        init_state, encoded_context = self.vln_bert('language', context_embs,)
        return init_state.unsqueeze(1), encoded_context, context_frame_embs  # (1, Dc), (1, Tc+1, Dc), (Tc, Dcf)
    
    def enc_step(self, context_embs, goal, state,):
        '''
        context_embs: (B, 1+Tc, Dc)
        goal: (B, H, W, C) / (B, Dg)
        state: (B, H, W, C)
        '''
        B = state.shape[0]
        # goal-state-action
        goal_repr = self.goal_encoder(goal) if len(goal.shape) > 2 else goal
        goal_emb = repeat(goal_repr, 'B D -> B N D',N=self.num_action)                            # (B, N, Dg)
        state_emb = repeat(self.state_encoder(state),'B D -> B N D',N=self.num_action)            # (B, N, Ds)
        action_emb = repeat(self.action_emb_layer(torch.arange(self.num_action).to(state.device)), 'N D -> B N D', B=B)  # (B, N, Da)
        gsa_emb = torch.cat((goal_emb, state_emb, action_emb), dim=-1)  # (B, N, Dg+Ds+Da)

        # h_t: (B, Dc), logit: (B, Tv), attended_language: (B, Dc), attended_visual: (B, Dc')
        h_t, logit, attended_language, attended_visual = self.vln_bert('visual', context_embs, gsa_emb)
        # update agent's state, unify history, language and vision by elementwise product
        vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)  # (B, 768)
        state_output = torch.cat((h_t, vis_lang_feat), dim=-1)                        # (B, 768*2)
        state_proj = self.state_LayerNorm(self.state_proj(state_output))              # (B, 768)

        return state_proj.unsqueeze(1), logit  # (B, 1, 768), (B, Tv)

    def forward(self, mode, context_embs, goal, state,):
        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, context_embs,)
            return init_state, encoded_sentence

        elif mode == 'visual':
            B = state.shape[0]
            # goal-state-action
            goal_emb = repeat(self.goal_encoder(goal), 'B D -> B N D',N=self.num_action)              # (B, N, Dg)
            state_emb = repeat(self.state_encoder(state),'B D -> B N D',N=self.num_action)            # (B, N, Ds)
            action_emb = repeat(self.action_emb_layer(torch.arange(self.num_action).to(state.device)), 'N D -> B N D', B=B)  # (B, N, Da)
            gsa_emb = torch.cat((goal_emb, state_emb, action_emb), dim=-1)  # (B, N, Dg+Ds+Da)

            # state_action_embed = torch.cat((sentence[:,0,:], action_feats), 1)  # (B, 768+4)
            # state_with_action = self.action_state_project(state_action_embed)   # (B, 768+4) -> (B, 768)
            # state_with_action = self.action_LayerNorm(state_with_action)        # (B, 768)
            # state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:,1:,:]), dim=1)  # (B, T, 768)
            # cand_feats[..., :-self.angle_feat_size] = self.drop_env(cand_feats[..., :-self.angle_feat_size])  # 

            # logit is the attention scores over the candidate features
            # h_t: (B, Dc), logit: (B, Tv), attended_language: (B, Dc), attended_visual: (B, Dc')
            h_t, logit, attended_language, attended_visual = self.vln_bert(mode, context_embs, gsa_emb)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)  # (B, 768)
            state_output = torch.cat((h_t, vis_lang_feat), dim=-1)                        # (B, 768*2)
            state_proj = self.state_LayerNorm(self.state_proj(state_output))              # (B, 768)

            return state_proj, logit  # (B, 768), (B, Tv)
        
    def polyak_target_update(self,tau=0.005):
        for param, target_param in zip(self.critic_head.parameters(), self.target_critic_head.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def visualize_emb(self, video):
        '''(1*T, H, W, C) -> (T, emb_dim)'''
        target_layers=self.state_encoder.net[-2:-1]#[self.state_encoder.net[-2][-1].conv2]
        # notice: input x should be (B, C, H, W) [0,1] cuda Tensor
        with EigenGradCAM(model=nn.Sequential(self.state_encoder, self.temporal_predictor), target_layers=target_layers) as cam:
            cam_masks = cam(input_tensor=video)
            cam_images = [show_cam_on_image(img, mask, use_rgb=True) for img, mask in zip(video.detach().cpu().numpy()/255.,cam_masks)]
        return cam_images


class Pi(VNBERTPolicy):
    def __init__(self, dataset, ckpt_file='', num_context_sample=900, mode='Q',context_type='SA',**kwargs):
        super(Pi, self).__init__(temporal_net=kwargs['temporal_net'])
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_type = context_type
        self.num_context_sample = num_context_sample
        self.action_threshold = 0.5
        self.video_frames, self.actions, self.goal_lists = dataset
        self.actions = torch.LongTensor(self.actions)
        if ckpt_file:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            ckpt = {k.replace('module.','') if 'module.' in k else k:v for k,v in ckpt.items()}
            self.load_state_dict(ckpt,strict=True)
            print(f'Load checkpoint from {ckpt_file}')
        else:
            raise ValueError('Please provide a checkpoint file!')
        
    def reset(self, ):
        # pre-compute fixed context embeddings using trained state encoder
        with torch.no_grad():
            if 'S' in self.context_type:
                selected_idx = remove_similar_frames(self.video_frames, self.num_context_sample)
                self.context_frames = torch.from_numpy(self.video_frames[selected_idx]).to(self.device)
                self.context_actions = self.actions[selected_idx].to(self.device) if self.context_type == 'SA' else None
                self.st, enc_context, _ = self.enc_context(self.context_frames, self.context_actions)  # (1, Dc), (1, Tc+1, Dc)
                self.enc_pure_context = enc_context[:,1:,:]  # (1, Tc, Dc)
            elif self.context_type == 'None':
                self.st = torch.zeros(1, 1, self.hidden_size).to(self.device)
                self.enc_pure_context = torch.zeros(1, self.num_context_sample, self.hidden_size).to(self.device)
        self.goal_emb = None
        self.stored_actions = dict(last_action=-1,duration=-1,action_prob=0)


    def act(self, observation_dict, timestep):
        if self.goal_emb is None:
            self.goal_emb = self.goal_encoder(torch.from_numpy(observation_dict['goal']).to(self.device).unsqueeze(0))
        
        if self.stored_actions['duration'] == -1:  # generate a new action
            with torch.no_grad():
                enc_context = torch.cat((self.st, self.enc_pure_context), dim=1)
                self.st, logit = self.enc_step(enc_context, self.goal_emb, torch.from_numpy(observation_dict['rgb'].copy()).to(self.device).unsqueeze(0))
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

if __name__ == "__main__":
    num_action = 9
    H, W, C = 224, 224, 3
    B, Tc, Tv = 8, 1+100, num_action

    # Initialize the VLNBERT model
    model = VNBERTPolicy().cuda()

    # Testing data for 'language' mode
    context_frames = torch.randint(0, 256, (Tc, H, W, C))  # -> (B, Tc, D)
    context_actions = torch.randint(0, num_action, (Tc,))

    init_state, encoded_sentence = model('language', (context_frames.cuda(), context_actions.cuda()),goal=None, state=None)
    print("Language mode output:")
    print("init_state shape:", init_state.shape)        # (B, 768)
    print("encoded_sentence shape:", encoded_sentence.shape)  # (B, Tc, 768)

    # Testing data for 'visual' mode
    context_frames = torch.randint(0, 256, (Tc, H, W, C))
    context_actions = torch.randint(0, num_action, (Tc,))
    goal = torch.randint(0, 256, (B, H, W, C))  
    state = torch.randint(0, 256, (B, H, W, C))     # -> (B, Tv, Dg+Ds+Da)

    state_proj, logit = model('visual', (context_frames.cuda(), context_actions.cuda()), goal.cuda(), state.cuda())
    print("Visual mode output:")
    print("state_proj shape:", state_proj.shape)  # (B, 768)
    print("logit shape:", logit.shape)            # (B, Tv)
    