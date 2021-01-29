from pose.models.hourglass import HourglassNet, Bottleneck
import cv2 
import numpy as np 
import torch
from PIL import Image
from matplotlib import cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HourglassNet(Bottleneck, num_stacks=2,num_blocks=1)
net = torch.nn.DataParallel(net).to(device)



net_weights = torch.load("./weights/model_best.pth.tar", map_location=torch.device('cpu'))

net.load_state_dict(net_weights['state_dict'])

img_name='./baseball.jpeg'
image = cv2.imread(img_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 画像のリサイズ
size = (256, 256)
img = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
# 画像の前処理
img = img.astype(np.float32) / 255.
# 色情報の標準化
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

# （高さ、幅、色）→（色、高さ、幅）
img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
# 画像をTensorに
img = torch.from_numpy(img)
# ミニバッチ化：torch.Size([1, 3, 368, 368])
x = img.unsqueeze(0)

#OpenPoseでheatmap, PAFs求める
net.eval()
output = net(x)
import numpy as np
output = np.array(output)
print(output)