#!/user/bin/python3
# coding=utf-8
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import matplotlib
matplotlib.use('Agg')
from src.data_loader import prepare_image_PIL
from src.models import RCF
from utils import load_vgg16pretrain
from os.path import join, isdir


def main(image_path):

    model = RCF()
    # print(model)
    model.cuda()
    model.apply(weights_init)
    load_vgg16pretrain(model, vggmodel="/root/workspace/repositories/RCF_Pytorch_Updated/model/vgg16convs.mat")

    img = np.array(Image.open(image_path), dtype=np.float32)
    img = prepare_image_PIL(img)
    test(model, "/root/workspace/repositories/RCF_Pytorch_Updated/out", img)

def test(model, save_dir, image):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    image = torch.from_numpy(image).cuda().unsqueeze(dim=0)
    image.cuda()
    _, _, H, W = image.shape
    results = model(image)
    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    results_all = torch.zeros((len(results), 1, H, W))
    for i in range(len(results)):
      results_all[i, 0, :, :] = results[i]
    torchvision.utils.save_image(1-results_all, join(save_dir, "%s.jpg" % "test"))
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save(join(save_dir, "%s.png" % "test"))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2) # as per https://github.com/yun-liu/rcf
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main(image_path="/root/workspace/repositories/RCF_Pytorch_Updated/results/8068.jpg")
