import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio
import librosa
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceNet(nn.Module):
    # Audio-visual CRW Net
    def __init__(self, args, pr, pretrained):
        super(FaceNet, self).__init__(args, pr, pretrained)
        self.net = InceptionResnetV1(pretrained=pretrained)
        self.net.last_linear = nn.Linear(self.net.last_linear.in_features, pr.feat_dim)
        self.net.last_bn = nn.Identity()
    
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.net(x)
        return x
