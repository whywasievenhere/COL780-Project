import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50
#TODO: Add necessary imports

class DummyModel(nn.Module):
    def __init__(self, batch_size, H, W, D):
        super(DummyModel, self).__init__()
        self.H = H
        self.W = W
        self.D = D
        self.B = batch_size

    def forward(self, x):
        out = torch.rand((self.B, self.D, self.H, self.W))
        return out

# TODO: Define your model architecture
        

class ReidModel(nn.Module):
    
    def __init__(self, local_conv_out_channels=128, num_classes=None):
        
        super().__init__()
        self.base = resnet50(pretrained=True)
        planes = 2048
        self.local_conv = nn.Conv2d(planes,local_conv_out_channels,1)
        self.local_norm = nn.BatchNorm2d(local_conv_out_channels)
        
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            nn.init.normal(self.fc.weight, std=0.001)
            nn.init.constant(self.fc.bias, 0)


    def forward(self, x):
        # shape [N, C, H, W]
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = F.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        if hasattr(self, 'fc'):
            logits = self.fc(global_feat)
            return global_feat, local_feat, logits

        return global_feat, local_feat