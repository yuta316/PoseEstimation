import torch
import torch.nn as nn
from torch.nn import init
import torchvision


class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()
        #1.Featureモジュール(vgg19)
        self.feature = Feature()

        #2.stage1
        self.stage1_p = makeOpenPoseBlock('block1_p')
        self.stage1_h = makeOpenPoseBlock('block1_h')

        #3.stage2
        self.stage2_p = makeOpenPoseBlock('block2_p')
        self.stage2_h = makeOpenPoseBlock('block2_h')

        #4.stage3
        self.stage3_p = makeOpenPoseBlock('block3_p')
        self.stage3_h = makeOpenPoseBlock('block3_h')

        #5.stage4
        self.stage4_p = makeOpenPoseBlock('block4_p')
        self.stage4_h = makeOpenPoseBlock('block4_h')

        #6.stage5
        self.stage5_p = makeOpenPoseBlock('block5_p')
        self.stage5_h = makeOpenPoseBlock('block5_h')

        #2.stage6
        self.stage6_p = makeOpenPoseBlock('block6_p')
        self.stage6_h = makeOpenPoseBlock('block6_h')

    def forward(self, x):

        #1.Featureモジュール(vgg19)
        out1 = self.feature(x)

        #2. stage1
        #PAFs
        out1_p = self.stage1_p(out1)
        #heatmap
        out1_h = self.stage1_h(out1)
        #3. stage2
        out2 = torch.cat([out1_p, out1_h, out1], 1)
        out2_p = self.stage2_p(out2)
        out2_h = self.stage2_h(out2)
        #4. stage3
        out3 = torch.cat([out2_p, out2_h, out1], 1)
        out3_p = self.stage3_p(out3)
        out3_h = self.stage3_h(out3)
        #5. stage4
        out4 = torch.cat([out3_p, out3_h, out1], 1)
        out4_p = self.stage4_p(out4)
        out4_h = self.stage4_h(out4)
        #6. stage5
        out5 = torch.cat([out4_p, out4_h, out1], 1)
        out5_p = self.stage5_p(out5)
        out5_h = self.stage5_h(out5)
        #2. stage6
        out6 = torch.cat([out5_p, out5_h, out1], 1)
        out6_p = self.stage6_p(out6)
        out6_h = self.stage6_h(out6)

        #ロス格納
        losses = []
        losses.append(out1_p)  # PAFs側
        losses.append(out1_h)  # confidence heatmap側
        losses.append(out2_p)
        losses.append(out2_h)
        losses.append(out3_p)
        losses.append(out3_h)
        losses.append(out4_p)
        losses.append(out4_h)
        losses.append(out5_p)
        losses.append(out5_h)
        losses.append(out6_p)
        losses.append(out6_h)

        return (out6_p, out6_h), losses

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        #VGG19を採用
        vgg19 = torchvision.models.vgg19(pretrained=True)
        model={}
        #使用するのは最初の10個の畳み込みまで
        model['feature'] = vgg19.features[0:23]
        #print(model)
        """
        {'feature': Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (17): ReLU(inplace=True)
        (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        )}
        """

        #畳み込み層を二つ追加
        model['feature'].add_module("23", torch.nn.Conv2d(512,256,kernel_size=3
            ,stride=1, padding=1))
        model['feature'].add_module("24", torch.nn.ReLU(inplace=True))
        model['feature'].add_module("25", torch.nn.Conv2d(256,128,kernel_size=3
            ,stride=1, padding=1))
        model['feature'].add_module("26", torch.nn.ReLU(inplace=True))

        self.model = model['feature']
        """
        (23): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (24): ReLU(inplace=True)
        (25): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (26): ReLU(inplace=True)
        )}
        """

    def forward(self, x):
        out = self.model(x)
        return out

def makeOpenPoseBlock(block_position):
    blocks = {}

    #config用意(別ファイルで用意する方がbetter)
    #stage1 
    #PASは入力128*128->38*38
    blocks['block1_p'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]
    #heatmapは入力128*128->19*19
    blocks['block1_h'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    #stage2~6
    for i in range(2, 7):
        #PASは入力128*128->38*38
        blocks['block%d_p'%i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]} ]
        #heatmapは入力128*128->19*19
        blocks['block%d_h'%i] =  [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]} ]

    #引数から構成config取り出す
    block = blocks[block_position]
    layers=[]
    #layersリストを作成し,要素を格納
    for i  in range(len(block)):
        for key, val in block[i].items():
            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=val[0], stride=val[i],
                    padding=val[2])]
            else:
                conv2d = nn.Conv2d(in_channels=val[0], out_channels=val[1],
                    kernel_size=val[2], stride=val[3], padding=val[4])
                layers+=[conv2d, nn.ReLU(inplace=True)]
    net = nn.Sequential(*layers[:-1])

    #初期化関数の設定し、畳み込み層を初期化
    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    net.apply(_initialize_weights_norm)

    return net

                