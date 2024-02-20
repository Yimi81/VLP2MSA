import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import *
from transformers import AutoModel
import json
import numpy as np

from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from model.model_TVI import TextVideoIntegrationTransformer
from model.prompt import VideoTextPrompt
from model.model_IHT import InterFrameHybridTransformer
import clip
import einops
class VLP2MSA(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 prompt_params,
                 fusion_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.video_params = video_params
        self.text_params = text_params
        self.prompt_params = prompt_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()


        ## ####################################
        # freeze bert
        ## ####################################
        # for name, param in self.text_model.named_parameters():
        #     # Bert freezing customizations 
        #     if "encoder.layer" in name:
        #         layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
        #         if layer_num <= 8:
        #             param.requires_grad = False

        pretrained = video_params['pretrained']
        num_frames = video_params.get('num_frames', 4)

        if video_params['model'] == "CLIP":
            arch_config = video_params.get('arch_config', 'ViT-B/16')
            clip_model, _ = clip.load(arch_config, device=self.device, jit=False) 
            clip_state_dict = clip_model.state_dict()
            ## ####################################
            # freeze clip
            ## ####################################
            # for paramclip in clip_model.parameters():
            #     paramclip.requires_grad = False

            vit = "visual.proj" in clip_state_dict
            if vit:
                vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
                vision_layers = len([k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
                vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
                grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
                image_resolution = vision_patch_size * grid_size
            else:
                counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
                vision_layers = tuple(counts)
                vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
                output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
                vision_patch_size = None
                assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
                image_resolution = output_width * 32

            embed_dim = clip_state_dict["text_projection"].shape[1]
            context_length = clip_state_dict["positional_embedding"].shape[0]
            vision_heads = vision_width // 64

            fusion_params["max_position_embeddings"] = context_length

            model = InterFrameHybridTransformer(
                                    input_resolution=image_resolution,
                                    patch_size=vision_patch_size,
                                    width=vision_width,
                                    layers=video_params['layers'],
                                    heads=vision_heads,
                                    output_dim=embed_dim,
                                    num_frames=num_frames)

            state_dict = {}
            for key, val in clip_state_dict.items():
                if "visual" in key:
                    new_key = key[7:]
                    state_dict[new_key] = val.clone()

            # if load_checkpoint in ["", None]:
            #     model.load_state_dict(state_dict, strict=False)

        self.video_model = model
        
        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)


        self.proj_l = nn.Linear(768, 512)
        self.text_prompts_generator = VideoTextPrompt(layers=prompt_params['layers'], embed_dim=prompt_params['embed_dim'], alpha=prompt_params['alpha'],)

        proj_embed_dim = 256
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.vision_proj = nn.Linear(embed_dim, proj_embed_dim)
        self.text_proj = nn.Linear(embed_dim, proj_embed_dim)

        self.fusion_transformer = TextVideoIntegrationTransformer(fusion_params)

        self.out_layer = nn.Linear(512, 1)

    def set_device(self, device):
        self.device = device

    def forward(self, data, bTrain=False):
        text_data = data['text'] 
        video_data = data['video'] 

        text_embeddings, text_mask = self.compute_text(text_data)  
        text_embeddings = self.proj_l(text_embeddings) 

        video_embeddings, video_feat = self.compute_video(video_data) 

        text_features = text_embeddings + self.text_prompts_generator(text_embeddings, video_embeddings)
        text_feat = F.normalize(self.text_proj(text_features[:,0,:]),dim=-1)   

        text_features = text_features[:,0,:]
        text_features = text_features.unsqueeze(1)

        fusion_output = self.compute_fusion(text_features, video_embeddings, text_mask)

        output = self.out_layer(fusion_output)
        output = output.squeeze(-1)
        if bTrain:
            return output, video_feat, text_feat, self.temp
        else:
            return output


    def compute_text(self, text_data):
        # 8 18
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])["last_hidden_state"]
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        text_mask = text_data['attention_mask']
        return text_embeddings, text_mask

    def compute_video(self, video_data):
        if self.video_params['model'] == "CLIP":
            video_hidden = self.video_model(einops.rearrange(video_data, 'b t c h w -> (b t) c h w')) #torch.Size([16, 4, 3, 224, 224])
            video_embeddings = einops.rearrange(video_hidden, '(b t) c -> b t c', t=self.video_params["num_frames"])#torch.Size([64, 512])
        else:
            video_embeddings = self.video_model(video_data)

        video_feat = F.normalize(self.vision_proj(video_embeddings[:,0,:]),dim=-1)  
        return video_embeddings,video_feat
    
    def compute_fusion(self, text_embeddings, video_embeddings, text_mask=None):
        concat_features = torch.cat((text_embeddings, video_embeddings), dim=1)  # concatnate tokens and frames
        video_mask = torch.ones((video_embeddings.shape[0], video_embeddings.shape[1])).cuda()

        fusion_output, pooled_output = self.fusion_transformer(concat_features)

        return pooled_output

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def compute_similarity(a, b, a_mask=None, b_mask=None, style='single', eps=1e-8, return_raw=False, temp=0.5):
    if style == 'single':
        sim = sim_matrix(a, b, eps=eps)
        return sim, sim.t()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    pass
