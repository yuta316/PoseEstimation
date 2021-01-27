import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict

NUM_LIMBS = 38
NUM_JOINTS = 18

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, paf_classes=NUM_LIMBS*2, ht_classes=NUM_JOINTS+1):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_features=128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3,self.inplanes, kernel_size=7, stride=2, padding=3,bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_features, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        #Hourglass　Model　構築
        ch = self.num_features*block.expansion
        hg,res,fc,score_paf,score_ht,fc_,paf_score_,ht_score_ = [],[],[],[],[],[],[],[]
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_features,4))
            res.append(self._make_residual(block, self.num_features, num_blocks))
            fc.append(self._make_fc(ch,ch))
            score_paf.append(nn.Conv2d(ch, paf_classes, kernel_size=1, bias=True))
            score_ht.append(nn.Conv2d(ch, ht_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                paf_score_.append(nn.Conv2d(paf_classes, ch, kernel_size=1, bias=True))
                ht_score_.append(nn.Conv2d(ht_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score_ht = nn.ModuleList(score_ht)
        self.score_paf = nn.ModuleList(score_paf)        
        self.fc_ = nn.ModuleList(fc_)
        self.paf_score_ = nn.ModuleList(paf_score_)
        self.ht_score_ = nn.ModuleList(ht_score_)
        
        self._initialize_weights_norm() 
    
    def _make_residual(self, block, planes, blocks, stride=1):
        """
        残差ブロック作成
        
        """
        downsample=None
        if ((stride != 1) or (self.inplanes != planes*block.expansion)):
            downsample = nn.Sequential(
                #1x1Conv
                nn.Conv2d(self.inplanes, planes*block.expansion,
                kernel_size=1, stride=stride, bias=True)
            )
        #layer作成
        layers=[]
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn , self.relu)   

    def forward(self, x):
        losses=[]
        out = []
        #CBR
        x = self.relu(self.bn1(self.conv1(x)))
        #layer1
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        for i in range(self.num_stacks):
            
            y = self.fc[i](self.res[i](self.hg[i](x)))
            
            score_paf = self.score_paf[i](y)
            score_ht = self.score_ht[i](y)
            
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                paf_score_ = self.paf_score_[i](score_paf)
                ht_score_ = self.ht_score_[i](score_ht)
                x = x+fc_+paf_score_+ht_score_
        losses.append(score_paf)
        losses.append(score_ht)
        return (score_paf, score_ht), losses
    
    def _initialize_weights_norm(self):        
       for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.0) 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 2
    """example:
     (0): Bottleneck(
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.conv1(out)))
        out = self.relu(self.bn3(self.conv2(out)))
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

