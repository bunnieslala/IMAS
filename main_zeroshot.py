import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torchattacks
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from core_raw import Smooth
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import cv2
from PIL import Image
import multiprocessing
import torch.distributions as D
from mask_generator_module import TipAdapterMaskGenerator

class AdvancedMaskGenerator:
    """训练时使用的高效mask生成器 - 修复版本"""
    
    def __init__(self, clip_model, device='cuda'):
        self.clip_model = clip_model
        self.device = device
      
    def generate_gradient_mask(self, image_batch, target_logits, mask_ratio=0.3):
        # 确保输入tensor在正确的设备上并且需要梯度
        if not image_batch.requires_grad:
            image_batch = image_batch.clone().detach().requires_grad_(True)
        
        # 确保tensor在GPU上
        if image_batch.device != torch.device(self.device):
            image_batch = image_batch.to(self.device)
            
        try:
            # 重新编码图像以获得可微分的特征
            image_features = self.clip_model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 计算目标类别的损失
            if target_logits.device != image_features.device:
                target_logits = target_logits.to(image_features.device)
                
            target_class = target_logits.argmax(dim=-1)
            
            # 使用简化的损失函数
            loss = F.cross_entropy(target_logits, target_class)
            
            # 清零之前的梯度
            if image_batch.grad is not None:
                image_batch.grad.zero_()
                
            # 反向传播
            loss.backward(retain_graph=True)
            
            # 获取梯度
            if image_batch.grad is None:
                #print("Warning: No gradients computed, falling back to random mask")
                #print("无法获得梯度，只能使用随机mask")
                return self._generate_random_mask(image_batch, mask_ratio)
                
            gradients = image_batch.grad.data
            saliency = torch.abs(gradients).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # 生成mask
            B, _, H, W = saliency.shape
            masks = []
            
            for b in range(B):
                sal = saliency[b, 0]  # [H, W]
                
                # 检查saliency是否有效
                if sal.sum() == 0 or torch.isnan(sal).any():
                    # 如果saliency无效，使用随机mask
                    mask = torch.rand(H, W, device=sal.device) > (1 - mask_ratio)
                    #print("使用的是随机mask")
                    mask = mask.float()
                else:
                    threshold = torch.quantile(sal, 1 - mask_ratio)
                    mask = (sal > threshold).float()
                    
                masks.append(mask)
            
            return torch.stack(masks, dim=0).unsqueeze(1)  # [B, 1, H, W]
            
        except Exception as e:
            print(f"Gradient computation failed: {e}, using random mask")
            return self._generate_random_mask(image_batch, mask_ratio)
    
    def _generate_random_mask(self, image_batch, mask_ratio=0.3):
        """生成随机mask作为备选方案"""
        B, C, H, W = image_batch.shape
        masks = []
        
        for b in range(B):
            # 随机选择要mask的像素
            mask = torch.rand(H, W, device=image_batch.device) < mask_ratio
            masks.append(mask.float())
            
        return torch.stack(masks, dim=0).unsqueeze(1)  # [B, 1, H, W]

def PGD_attack(model, image, label, epsilon=0.8, alpha=0.1, iters=10):
    image = image.to(device)
    label = label.to(device)
    loss = nn.CrossEntropyLoss()

    ori_image = image.data

    for i in range(iters):
        image.requires_grad = True
        output = model(image)

        model.zero_grad()
        cost = loss(output, label).to(device)
        cost.backward()

        # 对抗样本 = 原始图像 + 扰动
        adv_image = image + alpha * image.grad.sign()
        # 限制扰动范围
        eta = torch.clamp(adv_image - ori_image, min=-epsilon, max=epsilon)
        # 进行下一轮的对抗样本生成
        image = torch.clamp(ori_image + eta, min=0, max=1).detach()

    return image
def Auto_attack(model, image, label, epsilon=8/255):
    """
    Standard AutoAttack implementation.
    Returns the adversarial image.
    """
    # 1. Initialize the attack
    # 'Linf' is the standard norm used in your PGD example (via epsilon clamping)
    #atk = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', n_classes=1,verbose=False)
    atk=torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    
    # 2. Run the attack
    # AutoAttack handles the device internally, but ensure inputs are on the same device as model
    adv_image = atk(image, label)
    
    return adv_image

class TipAdapterWrapper_aom(torch.nn.Module):
    def __init__(self, clip_model, cache_keys, cache_values, clip_weights,adapter, beta=5.0, alpha=0.5,a=1.25):
        super().__init__()
        self.clip_model = clip_model
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.clip_weights = clip_weights
        self.adapter = adapter
        self.beta = beta
        self.alpha = alpha
        self.a=a
       

       

    def forward(self, x_anchor,x_source):

        image_features=self.a*x_anchor+(1-self.a)*x_source
        image_features=image_features / image_features.norm(dim=-1, keepdim=True)
        tip_logits=100.0 * image_features @ self.clip_weights
        return tip_logits

def evaluate_model_attack_rs(model_wrapper, model_wrapper_aom,test_loader, epsilon=4/255, alpha=1/255, iters=10,
                             num_samples=10, noise_std=0.12,clip_model=None,dataset_size=10):
    model_wrapper.eval()
    total_correct = 0
    total_samples = 0
    total_correct_aom=0
    total_samples_aom=0
    total_correct_ars=0
    total_correct_ars_aom=0
    our_aggregated_logits=0
    total_correct_clean=0
    n0=100
    n=1000
    alpha1=0.001
    total_sigma=0.25
    first_query_budget_frac=torch.tensor(np.sqrt(0.25))
    advanced_mask_gen = AdvancedMaskGenerator(clip_model)
    for images, target in tqdm(test_loader, desc="Evaluating Attack with Random Smoothing"):
        images, target = images.to(device), target.to(device)
        images = images.clone().detach().requires_grad_(True)
        # 先对每个batch利用PGD产生对抗样本
        #adv_images = PGD_attack(model_wrapper, images, target, epsilon=1/255, alpha=1/255, iters=iters)
        #atk =torchattacks.AutoAttack()
        # To make AutoAttack faster, use version='rand' (runs APGD-CE and APGD-DLR) or manually specify attacks
        # standard version runs APGD-CE, APGD-T, FAB-T, Square which is very slow
        atk=torchattacks.AutoAttack(model_wrapper, norm='Linf', eps=1/255, version='rand', n_classes=dataset_size, seed=None, verbose=False)
        # If version='rand' is not available or you want it even faster, uncomment the following line:
        # atk.attacks_to_run = ['apgd-ce']
        adv_images=atk(images,target)
        clean_images = images.clone().detach().requires_grad_(True)
        # 对每个对抗样本，进行num_samples次噪声采样，并平均输出
        batch_size = images.shape[0]
        aggregated_logits = torch.zeros(batch_size,dataset_size).to(device)  # val类别数为1000
        aggregated_logits_ars=torch.zeros(batch_size,dataset_size).to(device)
        image_feature=clip_model.encode_image(adv_images)
        images_transformed_feature=torch.zeros_like(image_feature).to(device)
        adv_images_feature=torch.zeros_like(image_feature).to(device)
        aom_feature=torch.zeros_like(image_feature).to(device)
        ars_feature=torch.zeros_like(image_feature).to(device)
        clean_images_feature=torch.zeros_like(image_feature).to(device)
        our_clean_images_feaeture=torch.zeros_like(image_feature).to(device)
        for _ in range(num_samples):
            sigma_1 = total_sigma / first_query_budget_frac
            noise = torch.randn_like(adv_images) * sigma_1
            first_noisy_images = adv_images + noise
            clean_first_noisy_images = clean_images + noise
            noise1=torch.randn_like(adv_images) * total_sigma
            noisy_images=adv_images+noise1
            noisy_images = torch.clamp(noisy_images, torch.min(noisy_images), torch.max(noisy_images))
            first_noisy_images = torch.clamp(first_noisy_images, torch.min(first_noisy_images), torch.max(first_noisy_images))
            if not first_noisy_images.requires_grad:
                first_noisy_images = first_noisy_images.clone().detach().requires_grad_(True)
            #print(f"first_noisy_images.requires_grad: {first_noisy_images.requires_grad}")
            logits = model_wrapper(first_noisy_images)
            clean_logits=model_wrapper(clean_first_noisy_images)
            mask_ratio = random.uniform(0.2, 0.5)
            #mask_ratio = 0.2
            masks = advanced_mask_gen.generate_gradient_mask(
                first_noisy_images, logits, mask_ratio
                )
            clean_masks = advanced_mask_gen.generate_gradient_mask(
                clean_first_noisy_images,clean_logits,  mask_ratio
                )
            with torch.no_grad():
                second_query_budget_frac = torch.sqrt(1 - torch.square(first_query_budget_frac))
                norm_2 = torch.sqrt(torch.tensor(3)) * torch.norm(masks.view(masks.shape[0], -1), p=2, dim=1)
                sigma_2 = (total_sigma /second_query_budget_frac) * (norm_2 / np.sqrt(3*224*224))
                sigma_2 = torch.maximum(torch.ones(sigma_2.shape).to(device) * 1e-6, sigma_2) 
                sigma_2 = sigma_2.view(-1, 1, 1, 1)
                clean_norm_2=torch.sqrt(torch.tensor(3)) * torch.norm(clean_masks.view(masks.shape[0], -1), p=2, dim=1)
                clean_sigma_2 = (total_sigma /second_query_budget_frac) * (clean_norm_2 / np.sqrt(3*224*224))
                clean_sigma_2 = torch.maximum(torch.ones(clean_sigma_2.shape).to(device) * 1e-6, clean_sigma_2)
                clean_sigma_2 = clean_sigma_2.view(-1, 1, 1, 1)
                noise = torch.randn_like(images, device='cuda') * sigma_2.expand_as(images)
                clean_noise = torch.randn_like(images, device='cuda') * clean_sigma_2.expand_as(images)
                images_transformed = torch.mul(adv_images+noise, masks)
                our_clean_images = torch.mul(clean_images+clean_noise, clean_masks)
                pre_averaging_x_transformed = images_transformed.clone().detach()
                sigma_1 = sigma_1.repeat(sigma_2.shape[0]).reshape(sigma_2.shape[0],1,1,1).repeat(1,1,224,224).to(device)
                sigma_2 = sigma_2.reshape(sigma_2.shape[0],1,1,1).repeat(1,1,224,224).to(device)
                clean_sigma_2=clean_sigma_2.reshape(clean_sigma_2.shape[0],1,1,1).repeat(1,1,224,224).to(device)
                denominator = ((masks ** 2) * (sigma_1 ** 2)) + (sigma_2 ** 2).to(device)
                clean_denominator = ((clean_masks ** 2) * (sigma_1 ** 2)) + (clean_sigma_2 ** 2).to(device)
                w1 = sigma_2 ** 2 / denominator
                w2 = ((sigma_1 ** 2) * (masks ** 2)) / denominator
                clean_w1=clean_sigma_2 ** 2 / clean_denominator
                clean_w2=((sigma_1 ** 2) * (clean_masks ** 2)) / clean_denominator
                images_transformed *= w2
                our_clean_images *=clean_w2
                our_clean_images+=(clean_w1*clean_images)
                images_transformed_2=images_transformed.clone().detach().requires_grad_(True)
                #images_transformed_2*=w2
                images_transformed += (w1 * adv_images)
                #images_transformed +=torch.mul(adv_images, 1-masks)

                images_transformed_2+=(w1 * first_noisy_images)
                images_transformed_feature+=clip_model.encode_image(images_transformed)
                adv_images_feature+=clip_model.encode_image(adv_images)
                clean_images_feature+=clip_model.encode_image(clean_images)
                aom_feature+=clip_model.encode_image(noisy_images)
                ars_feature+=clip_model.encode_image(images_transformed_2)
                logits = model_wrapper(images_transformed_2)
                our_clean_images_feaeture+=clip_model.encode_image(our_clean_images)
                #our_logits = model_wrapper(images_transformed)
            aggregated_logits_ars += logits
            #our_aggregated_logits+=our_logits
        images_transformed_feature/=num_samples
        adv_images_feature/=num_samples
        aom_feature/=num_samples
        aggregated_logits_ars /= num_samples
        ars_feature/=num_samples
        our_clean_images_feaeture/=num_samples
        clean_images_feature/=num_samples
        #our_aggregated_logits/=num_samples
        aggregated_logits=model_wrapper_aom(images_transformed_feature,adv_images_feature)
        aom_logits=model_wrapper_aom(aom_feature,adv_images_feature)
        ars_aom_logits=model_wrapper_aom(ars_feature,adv_images_feature)
        our_clean_images_logits=model_wrapper_aom(our_clean_images_feaeture,clean_images_feature)
        clean_preds = our_clean_images_logits.argmax(dim=1)
        preds = aggregated_logits.argmax(dim=1)
        preds_aom=aom_logits.argmax(dim=1)
        preds_ars=aggregated_logits_ars.argmax(dim=1)
        preds_ars_aom=ars_aom_logits.argmax(dim=1)
        #preds_our=our_aggregated_logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)
        total_correct_aom += (preds_aom == target).sum().item()
        total_correct_clean += (clean_preds == target).sum().item()
        total_correct_ars+=(preds_ars == target).sum().item()
        total_correct_ars_aom+=(preds_ars_aom == target).sum().item()
        #total_correct_our+=(preds_our == target).sum().item()
        total_samples_aom += target.size(0)
    acc = 100. * total_correct / total_samples
    aom_acc=100.*total_correct_aom/total_samples_aom
    ars_acc=100.*total_correct_ars/total_samples_aom
    ars_aom_acc=100.*total_correct_ars_aom/total_samples_aom
    clean_acc =100. * total_correct_clean / total_samples
    #our_acc=100.*total_correct_our/total_samples_aom
    return acc,aom_acc,ars_acc,ars_aom_acc,clean_acc
def evaluate_model_attack(model_wrapper, test_loader, epsilon=0.8, alpha=0.1, iters=10):
    model_wrapper.eval()
    total_correct = 0
    total_samples = 0

    for images, target in tqdm(test_loader, desc="Evaluating Attack"):
        images, target = images.to(device), target.to(device)
        images = images.clone().detach().requires_grad_(True)
        # 对每个batch利用PGD_attack生成对抗样本
        adv_images = PGD_attack(model_wrapper, images, target, epsilon=epsilon, alpha=alpha, iters=iters)
        # 生成的对抗样本用于前向传播得到预测得分
        with torch.no_grad():
            outputs = model_wrapper(adv_images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)
    acc = 100. * total_correct / total_samples
    return acc
class ZeroShotClassifier(nn.Module):
    def __init__(self, clip_model, clip_weights):
        super(ZeroShotClassifier, self).__init__()
        self.clip_model = clip_model
        self.clip_weights = clip_weights.to('cuda',dtype=torch.float32)  

    def forward(self, image):
        features = self.clip_model.encode_image(image).to('cuda',dtype=torch.float32)  
        features = features / features.norm(dim=-1, keepdim=True)
        logits = 100. * features @ self.clip_weights
        return logits
def main():
    """主函数 - 带mask增强的训练"""
    #multiprocessing.set_start_method('spawn')
    # 加载配置
    cfg = yaml.load(open("/home/guanbeibei/Tip-Adapter-main/Tip-Adapter-main/configs/food101.yaml", 'r'), Loader=yaml.Loader)
    
    # 添加mask增强相关配置
    mask_config = {
        'use_mask_augment': True,
        'mask_augment_prob': 0.5,  # 使用mask增强的概率
        'mask_loss_weight': 0.1,   # mask一致性损失权重
        'use_contrastive_loss': False,
        'contrastive_loss_weight': 0.05,
        'mask_strategies': ['random_patches', 'saliency_based'],
        'mask_prob': 0.5,  # 数据集中应用mask的概率
        'shots':1
    }
    cfg.update(mask_config)
    
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    
    print("\nRunning configs with mask augmentation.")
    print(cfg, "\n")
    
    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    
    # 准备数据集
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset with mask augmentation.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, 
                                  tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, 
                                   tfm=preprocess, shuffle=False)
    
    # 训练时的变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), 
                                   interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                           std=(0.26862954, 0.26130258, 0.27577711))
    ])
    
    # 创建cache加载器（不使用mask增强）
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, 
                                          tfm=train_transform, is_train=True, shuffle=False)
    
    # 文本特征
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    
    # 构建cache模型
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)
    
    # 预加载测试和验证特征
    test_features, test_labels = pre_load_features(cfg, "test", cache_keys, cache_values,
                                                  clip_model, test_loader, clip_weights)
    val_features, val_labels = pre_load_features(cfg, "val", cache_keys, cache_values,
                                                clip_model, val_loader, clip_weights)
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    model_wrapper_aom=TipAdapterWrapper_aom(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=adapter,
            beta=cfg['init_beta'],
            alpha=cfg['init_alpha'],
            a=1.25
        ).cuda()
    zero_shot_net= ZeroShotClassifier(clip_model, clip_weights).to(device)
    attack_epsilon = 1/255  # 可根据需求调整
    attack_alpha = 1/255  # 可根据需求调整
    attack_iters = 100
    num_class=dataset.num_classes
    #adv_acc=evaluate_model_attack(base_classifier, test_loader, epsilon= 1/255, alpha=1/255, iters=100)
    #print("adv_acc:{:.2f}%".format(adv_acc))
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    acc_clip,aom_acc,ars_acc,ars_aom_acc,clean_acc = evaluate_model_attack_rs(zero_shot_net,model_wrapper_aom, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=10, noise_std=0.25,clip_model=clip_model,dataset_size=num_class)
    print("Zero-shot CLIP Accuracy under PGD attack: {:.2f}%".format(acc_clip))
    print("aom Accuracy under PGD attack: {:.2f}%".format(aom_acc))
    print("ars Accuracy under PGD attack: {:.2f}%".format(ars_acc))
    print("ars_aom Accuracy under PGD attack: {:.2f}%".format(ars_aom_acc))
    print("clean Accuracy under PGD attack: {:.2f}%".format(clean_acc))
    '''
    print("Evaluating Zero-shot CLIP without defense under PGD Attack ...")
    acc=evaluate_model_attack(base_classifier, test_loader, epsilon=1/255, alpha=1/255, iters=100)
    print("Accuracy under PGD attack: {:.2f}%".format(acc))
    '''
    #print("our Accuracy under PGD attack: {:.2f}%".format(our_acc))
    '''
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    acc_clip,radius = evaluate_model_rs(base_classifier, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=10, noise_std=0.12,num_classes=clip_weights.size(0))
    print("Zero-shot CLIP Accuracy under PGD attack: {:.2f}%".format(acc_clip))
    
    print("Zero-shot CLIP radius under PGD attack: {:.2f}%".format(radius))
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    acc_clip,radius = evaluate_model_rs(base_classifier, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=10, noise_std=0.25,num_classes=clip_weights.size(0))
    print("Zero-shot CLIP Accuracy under PGD attack: {:.2f}%".format(acc_clip))
    print("Zero-shot CLIP radius under PGD attack: {:.2f}%".format(radius))    
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    acc_clip,radius = evaluate_model_rs(base_classifier, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=10, noise_std=0.5,num_classes=clip_weights.size(0))
    print("Zero-shot CLIP Accuracy under PGD attack: {:.2f}%".format(acc_clip))
    print("Zero-shot CLIP radius under PGD attack: {:.2f}%".format(radius))
    '''

def cls_acc(output, target, topk=1):
    """计算分类准确率"""
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    """构建CLIP分类器"""
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


if __name__ == '__main__':
    main()