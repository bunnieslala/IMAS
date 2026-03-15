import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets
import torch
import cv2


template = ['a photo of a {}.']

import cv2
import numpy as np
from PIL import Image

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    #image= np.array(image)
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=7):
    """PGD攻击（PyTorch版本）"""
    # 初始化对抗样本：原始图像 + 随机噪声
    adv_images = images.clone().detach()
    adv_images += torch.randn_like(images) * epsilon  # 随机初始化扰动
    
    for _ in range(num_iter):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        # 更新对抗样本：沿梯度符号方向添加扰动
        adv_images = adv_images.detach() + alpha * grad.sign()
        # 投影到ε约束范围内
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, 0, 1).detach_()
    
    return adv_images

from pathlib import Path


    


    
class Caltech101(DatasetBase):

    dataset_dir = '/data0/tongbs/datasets/classification/caltech-101'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)


        super().__init__(train_x=train, val=val, test=test)