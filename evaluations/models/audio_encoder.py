import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models
import torchaudio
import librosa

import sys
sys.path.append('..')
from utils import sourcesep
from config import params
import models



# -------------------------------------------------------------------------------------- # 


class AudioEncoder(nn.Module):
    # base encoder
    def __init__(self, args, pr, device=None):
        super(AudioEncoder, self).__init__()
        self.pr = pr
        self.args = args
        self.num_classes = pr.num_classes
        self.wav2spec = args.wav2spec
        self.trans = self.stft_transform()
    
    def stft_transform(self):
        win_length = 256
        n_fft = self.pr.n_fft

        if self.pr.clip_length in [0.96, 2.55]:
            hop_length = 160
        else:
            sample_num = int(self.pr.clip_length * self.pr.samp_sr)
            if self.args.finer_hop:
                hop_length = int(sample_num // 256) 
            else:
                hop_length = int(sample_num // 128) 
        
        trans = torchaudio.transforms.Spectrogram(
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft, 
            power=None
        )
        return trans

    def unfold2patch(self, audio):
        '''
            audio shape: (N, 1, L)
        '''
        # import pdb; pdb.set_trace()
        patch_size = int(self.pr.clip_length * self.pr.samp_sr)
        # audio shape: (N, 1, patch_num，patch_size)
        audio = audio.unfold(-1, patch_size, self.pr.patch_stride)
        # audio shape: (N, patch_num, 1, patch_size)
        audio = audio.permute(0, 2, 1, 3)
        return audio
    

    def wav2stft(self, audio):
        # import pdb; pdb.set_trace()
        '''
            waveform shape: (N, K, 1, L)
        '''
        audio = audio.squeeze(-2)
        audio_size = audio.size()
        # audio = audio.contiguous().view(audio_size[0] * audio_size[1], *audio_size[2:])
        spec = self.trans(audio)
        # spec = spec.view(audio_size[0], audio_size[1], *spec.shape[1:])
        spec = spec.permute(0, 1, 4, 3, 2)[..., :-1, :-1]
        return spec

# -------------------------------------------------------------------------------------- # 

class WaveNet(AudioEncoder):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(WaveNet, self).__init__(args, pr, device)
        backbone = args.backbone if backbone is None else backbone
        # self.backbone = backbone
        self.wav2spec = args.wav2spec
        if backbone in ['resnet9']:
            in_channels = 2
            model = torchvision.models.resnet._resnet('resnet9', torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], pretrained=False, progress=False)
            model.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model.fc = nn.Linear(model.fc.in_features, pr.feat_dim)
            self.net = model
        elif backbone in ['resnet18']:
            in_channels = 2
            model = torchvision.models.resnet._resnet('resnet18', torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], pretrained=False, progress=False)
            model.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model.fc = nn.Linear(model.fc.in_features, pr.feat_dim)
            self.net = model
        
        
        self.criterion = models.__dict__[pr.loss](args, pr, device)
    
    def forward(self, inputs, evaluate=False, loss=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_left: (N, K, C, D) or (N, K, C, H, W)
            audio_right: (N, K, C, D) or (N, K, C, H, W)
        '''
        audio_left = inputs['left_audios']
        audio_right = inputs['right_audios']
        delay_time = inputs['delay_time']

        audio_left = self.encode_audio(audio_left)
        audio_right = self.encode_audio(audio_right)

        output = {
            'audio_left': audio_left,
            'audio_right': audio_right,
            'delay_time': delay_time
        }
        
        if evaluate:
            res = self.criterion.evaluate(output)
            return res
        if loss:
            loss = self.criterion(output).view(1, -1)
            return loss
        
        return output
    
    def encode_audio(self, audio):
        # import pdb; pdb.set_trace()
        audio = self.unfold2patch(audio)
        if self.args.wav2spec:
            audio = self.wav2stft(audio)
        audio_size = audio.shape
        audio = audio.contiguous().view(audio_size[0] * audio_size[1], *audio_size[2:])
        audio = self.net(audio)
        audio = audio.contiguous().view(*audio_size[:2], -1)
        return audio



class WaveAugNet(WaveNet):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(WaveAugNet, self).__init__(args, pr, device, backbone)
    
    def forward(self, inputs, evaluate=False, loss=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_left: (N, K, C, D) or (N, K, C, H, W)
            audio_right: (N, K, C, D) or (N, K, C, H, W)
            audio_noaug_left: (N, K, C, D) or (N, K, C, H, W)
        '''
        audio_left = inputs['left_audios']
        audio_right = inputs['right_audios']
        audio_noaug_left = inputs['noaug_left_audios']
        shift_offset = inputs['shift_offset']
        delay_time = inputs['delay_time']

        audio_left = self.encode_audio(audio_left)
        audio_right = self.encode_audio(audio_right)
        audio_noaug_left = self.encode_audio(audio_noaug_left)

        output = {
            'audio_left': audio_left,
            'audio_right': audio_right,
            'audio_noaug_left': audio_noaug_left,
            'shift_offset': shift_offset,
            'delay_time': delay_time
        }
        
        if evaluate:
            res = self.criterion.evaluate(output)
            return res
        if loss:
            loss = self.criterion(output).view(1, -1)
            return loss
        
        return output


class WaveMixNet(WaveNet):
    # Audio Relative Depth Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(WaveMixNet, self).__init__(args, pr, device, backbone)
    
    def forward(self, inputs, evaluate=False, loss=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_left: (N, K, C, D) or (N, K, C, H, W)
            audio_right: (N, K, C, D) or (N, K, C, H, W)
            audio_aug_left: (N, K, C, D) or (N, K, C, H, W)
            audio_neg_left: (N, K, C, D) or (N, K, C, H, W)

        '''
        audio_left = inputs['left_audios']
        audio_right = inputs['right_audios']
        audio_noaug_left = inputs['noaug_left_audios']
        audio_noaug_right = inputs['noaug_right_audios']

        shift_offset = inputs['shift_offset']
        delay_time = inputs['delay_time']

        audio_left = self.encode_audio(audio_left)
        audio_right = self.encode_audio(audio_right)
        audio_noaug_left = self.encode_audio(audio_noaug_left)
        audio_noaug_right = self.encode_audio(audio_noaug_right)

        output = {
            'audio_left': audio_left,
            'audio_right': audio_right,
            'audio_noaug_left': audio_noaug_left,
            'audio_noaug_right': audio_noaug_right,
            'shift_offset': shift_offset,
            'delay_time': delay_time
        }
        
        if evaluate:
            res = self.criterion.evaluate(output)
            return res
        if loss:
            loss = self.criterion(output).view(1, -1)
            return loss
        
        return output


