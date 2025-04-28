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
from tqdm import tqdm



def test_single_image(path, model):
    img_filename = path.split("/")[-1]
    img_directory = path.replace('/' + path.split("/")[-1], '')

    image = np.array(Image.open(join(img_filename, img_directory)), dtype=np.float32)
    img = np.dstack((image, np.zeros_like(image), np.zeros_like(image)))

    img = prepare_image_PIL(img)
    test(model, "/root/workspace/repositories/RCF_Pytorch_Updated/out", img, img_filename)


def test_image_folder(directory, model):
    images_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for img_filename in tqdm(images_files):
        path = join(directory, img_filename)
        image = np.array(Image.open(path), dtype=np.float32)
        #img = np.dstack((image, np.zeros_like(image), np.zeros_like(image)))

        img = prepare_image_PIL(image)
        test(model, "/root/workspace/repositories/RCF_Pytorch_Updated/out", img, img_filename)

def main(path):

    model = RCF()
    # print(model)
    model.cuda()
    model.apply(weights_init)
    load_vgg16pretrain(model, vggmodel="/root/workspace/repositories/RCF_Pytorch_Updated/model/vgg16convs.mat")

    checkpoint = torch.load("/root/workspace/repositories/RCF_Pytorch_Updated/model/checkpoints/checkpoint_epoch29.pth")
    model.load_state_dict(checkpoint['state_dict'])
    if os.path.isfile(path):
        test_single_image(path, model)
    elif os.path.isdir(path):
        test_image_folder(path, model)
    else:
        raise FileExistsError(f"the path provided: \n ->{path} \n-> is neither a file or a directory.")


def test(model, save_dir, image, img_filename):
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
    torchvision.utils.save_image(1-results_all, join(save_dir, "%s.jpg" % img_filename))
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save(join(save_dir, "%s.png" % img_filename))


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
    main(path="/root/workspace/repositories/RCF_Pytorch_Updated/data/test")
