import torch
import numpy as np
from torch import nn
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
from mtr.modules.head import CLIPHead
from config import CLUSTERS

class ContrastiveModel(nn.Module):
    def __init__(self, audio_encoder, text_encoder, text_type, audio_dim, text_dim, mlp_dim, temperature, disentangle, n_proj):
        super(ContrastiveModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.text_type = text_type
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad=True)
        self.head = CLIPHead(logit_scale=self.logit_scale)
        if disentangle:
            self.audio_projector = nn.ModuleDict()
            self.text_projector = nn.ModuleDict()
            for c in CLUSTERS:
                self.audio_projector[c] = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
                self.text_projector[c] = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, mlp_dim, bias=False))
                for i in range(n_proj - 1):
                    self.audio_projector[c].add_module(f'relu{i}', nn.ReLU())
                    self.audio_projector[c].add_module(f'linear{i}', nn.Linear(mlp_dim, mlp_dim, bias=False))
                    self.text_projector[c].add_module(f'relu{i}', nn.ReLU())
                    self.text_projector[c].add_module(f'linear{i}', nn.Linear(mlp_dim, mlp_dim, bias=False))
            if disentangle == 'audio':
                for c in CLUSTERS[1:]:
                    self.text_projector[c] = self.text_projector[CLUSTERS[0]]
        else:
            self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
            self.text_projector =  nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, mlp_dim, bias=False))
            for i in range(n_proj - 1):
                self.audio_projector.add_module(f'relu{i}', nn.ReLU())
                self.audio_projector.add_module(f'linear{i}', nn.Linear(mlp_dim, mlp_dim, bias=False))
                self.text_projector.add_module(f'relu{i}', nn.ReLU())
                self.text_projector.add_module(f'linear{i}', nn.Linear(mlp_dim, mlp_dim, bias=False))
        
        self.audio_encoder.train()
        self.text_encoder.train()
        self.a_latent = nn.Identity()
        self.t_latent = nn.Identity()
        self.disentangle = disentangle

    def forward(self, audio, text, text_mask=None, cluster_mask=None):
        h_audio = self.encode_audio(audio)
        if self.text_type == "bert":
            h_text = self.encode_bert_text(text, text_mask)
        elif self.text_type == "glove":
            h_text = self.encode_glove_text(text)
        audio_loss, audio_correct = self.head(h_audio, h_text, cluster_mask)
        text_loss, text_correct = self.head(h_text, h_audio, cluster_mask)
        loss = (audio_loss + text_loss) / 2
        return loss, audio_correct, text_correct, self.logit_scale
        
    def encode_audio(self, audio):
        # audio = (Batch x Length x Dim)
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        if self.disentangle:
            z_audio = torch.stack([self.audio_projector[c](h_audio) for c in CLUSTERS])
        else:
            z_audio = self.audio_projector(h_audio)
        return z_audio

    def encode_bert_text(self, text, text_mask):
        text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
        
        if self.disentangle:
            text_emb = text_emb['last_hidden_state'][:,0,:].view(len(CLUSTERS), -1, self.text_dim)
            h_text = self.t_latent(text_emb)
            z_text = torch.stack([self.text_projector[c](h_text[i]) for i, c in enumerate(CLUSTERS)])
        else:
            text_emb = text_emb['last_hidden_state'][:,0,:]
            h_text = self.t_latent(text_emb)
            z_text = self.text_projector(h_text)
        return z_text

    def encode_glove_text(self, text_emb): 
        h_text = self.t_latent(text_emb)
        z_text = self.text_projector(h_text)
        return z_text
    