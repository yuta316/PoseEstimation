import cv2 
import numpy as np 
import torch
from PIL import Image
from model import OpenPoseNet
from matplotlib import cm
#networkのインスタンス化
net = OpenPoseNet()

# 学習済みパラメータをロードする
net_weights = torch.load(
    './open_pose/weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}

# ロードした内容をモデルのパラメータ名net.state_dict().keys()にコピーする
for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]] = net_weights[list(keys)[i]]

# コピーした内容をモデルに与える
state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state, strict=False)

print('ネットワーク設定完了：学習済みの重みをロードしました')

img_name='./open_pose/image/baseball.jpeg'
image = cv2.imread(img_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 画像のリサイズ
size = (368, 368)
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
output, _ = net(x)

# 画像をテンソルからNumPyに変化し、サイズを戻します
pafs = output[0][0].detach().numpy().transpose(1, 2, 0)
heatmaps = output[1][0].detach().numpy().transpose(1, 2, 0)

pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

pafs = cv2.resize(
    pafs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(
    heatmaps, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

# 左肘と左手首のheatmap、そして左肘と左手首をつなぐPAFのxベクトルを可視化する
# 左肘
heat_map = heatmaps[:, :, 1]  # 6は左肘
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))
# 合成して表示
cv2.imwrite("open_pose/image/a.jpg",heat_map)
blend_img = cv2.addWeighted(image, 0.5, heat_map, 0.5, 0)
cv2.imwrite("open_pose/image/d.jpg",blend_img)

# 左手首
heat_map = heatmaps[:, :, 0]  # 7は左手首
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
heat_map = np.asarray(heat_map.convert('RGB'))
# 合成して表示
blend_img2 = cv2.addWeighted(image, 0.5, heat_map, 0.5, 0)
cv2.imwrite("open_pose/image/b.jpeg",blend_img2)


# 左肘と左手首をつなぐPAFのxベクトル
paf = pafs[:, :, 24]
paf = Image.fromarray(np.uint8(cm.jet(paf)*255))
paf = np.asarray(paf.convert('RGB'))
# 合成して表示
blend_img3= cv2.addWeighted(image, 0.5, paf, 0.5, 0)
cv2.imwrite("open_pose/image/c.jpg",blend_img3)