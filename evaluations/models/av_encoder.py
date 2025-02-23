import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio
import librosa
from facenet_pytorch import MTCNN, InceptionResnetV1

import sys
sys.path.append('..')
from utils import sourcesep, torch_utils
from config import params
# import models
from models import *


# -------------------------------------------------------------------------------------- # 

class AudioVisualCRWNet(AudioEncoder):
    # Audio-visual CRW Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(AudioVisualCRWNet, self).__init__(args, pr, device)
        self.img_feat_scaling = args.img_feat_scaling
        self.crop_face = args.crop_face
        backbone = args.backbone if backbone is None else backbone

        self.audio_net = self.load_audio_model(args, pr)
        if self.crop_face:
            model = InceptionResnetV1(pretrained='vggface2')
            model.last_linear = nn.Linear(model.last_linear.in_features, pr.feat_dim)
            model.last_bn = nn.Identity()
        else: 
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, pr.feat_dim)
        self.vision_net = model

        self.criterion = models.__dict__[pr.loss](args, pr, device)
    
    def forward(self, inputs, evaluate=False, loss=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio_left: (N, K, C, D) or (N, K, C, H, W)
            audio_right: (N, K, C, D) or (N, K, C, H, W)
        '''
        audio_left = inputs['left_audios']
        audio_right = inputs['right_audios']
        shift_offset = inputs['shift_offset']
        delay_time = inputs['delay_time']
        img = inputs['img']

        audio_left = self.forward_audio_with_img(audio_left, img)
        audio_right = self.forward_audio_with_img(audio_right, img)

        output = {
            'audio_left': audio_left,
            'audio_right': audio_right,
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

    def forward_audio_with_img(self, audio, img):
        # import pdb; pdb.set_trace()
        audio = self.unfold2patch(audio)
        if self.args.wav2spec:
            audio = self.wav2stft(audio)
        audio_size = audio.shape
        audio = audio.contiguous().view(audio_size[0] * audio_size[1], *audio_size[2:])

        audio = self.audio_net.conv1(audio)
        audio = self.audio_net.bn1(audio)
        audio = self.audio_net.relu(audio)
        audio = self.audio_net.maxpool(audio)
        audio = self.audio_net.layer1(audio)
        audio = self.audio_net.layer2(audio)

        feat_img = self.vision_net(img)
        feat_img = feat_img.unsqueeze(1).expand(-1, audio_size[1], -1)
        feat_img = feat_img.contiguous().view(-1, feat_img.size(-1), 1, 1)
        feat_img = feat_img.expand(-1, -1, audio.shape[-2], audio.shape[-1])
        img_feat_scaling = 1 / (audio.shape[-2] * audio.shape[-1])  if self.img_feat_scaling is None else self.img_feat_scaling
        audio = torch.cat([audio, feat_img * img_feat_scaling], dim=1)
        audio = self.audio_net.layer3(audio)
        audio = self.audio_net.layer4(audio)

        audio = self.audio_net.avgpool(audio)
        audio = audio.view(audio.size(0), -1)
        audio = self.audio_net.fc(audio)
        audio = audio.contiguous().view(*audio_size[:2], -1)
        return audio

    def load_audio_model(self, args, pr):
        model = WaveNet(args, pr)
        if args.pretrained:
            resume = './checkpoints/' + args.pretrained
            model, _ = torch_utils.load_model(resume, model, strict=True)
        net = model.net 
        net.inplanes = 256
        net.layer3 = net._make_layer(torchvision.models.resnet.BasicBlock, 256, 1, stride=2)
        return net


class AudioVisualAugCRWNet(AudioVisualCRWNet):
    # Audio-visual CRW Net
    def __init__(self, args, pr, device=None, backbone=None):
        super(AudioVisualAugCRWNet, self).__init__(args, pr, device)
    
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
        img = inputs['img']

        audio_left = self.forward_audio_with_img(audio_left, img)
        audio_right = self.forward_audio_with_img(audio_right, img)
        audio_noaug_left = self.forward_audio_with_img(audio_noaug_left, img)

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

