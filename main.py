import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import copy
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
import torchattacks
class MaskAugmentedDataset(Dataset):
    """支持mask增强的数据集"""
    
    def __init__(self, data_source, transform, mask_generator=None, mask_prob=0.5, 
                 mask_strategies=['random_patches', 'saliency_based', 'mixed']):
        self.data_source = data_source
        self.transform = transform
        self.mask_generator = mask_generator
        self.mask_prob = mask_prob
        self.mask_strategies = mask_strategies
        
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = item.impath
        label = item.label
        
        # 加载图像
        image = Image.open(image).convert('RGB')
        
        # 应用基础变换
        if self.transform:
            image_tensor = self.transform(image)
        
        # 根据概率决定是否应用mask
        if random.random() < self.mask_prob and self.mask_generator is not None:
            # 随机选择mask策略
            strategy = random.choice(self.mask_strategies)
            image_tensor = self._apply_mask_strategy(image_tensor, strategy)
        
        return image_tensor, label
    
    def _apply_mask_strategy(self, image_tensor, strategy):
        """应用不同的mask策略"""
        if strategy == 'random_patches':
            return self._random_patch_mask(image_tensor)
        elif strategy == 'saliency_based':
            return self._saliency_based_mask(image_tensor)
        elif strategy == 'mixed':
            # 随机选择一种方法
            sub_strategy = random.choice(['random_patches', 'saliency_based'])
            return self._apply_mask_strategy(image_tensor, sub_strategy)
        else:
            return image_tensor
    
    def _random_patch_mask(self, image_tensor, num_patches=3, patch_size_range=(16, 64)):
        """随机遮挡补丁"""
        C, H, W = image_tensor.shape
        masked_image = image_tensor.clone()
        
        for _ in range(num_patches):
            # 随机补丁大小
            patch_h = random.randint(*patch_size_range)
            patch_w = random.randint(*patch_size_range)
            
            # 随机位置
            start_h = random.randint(0, max(0, H - patch_h))
            start_w = random.randint(0, max(0, W - patch_w))
            
            # 随机遮挡值
            mask_value = random.choice([0.0, torch.randn(C, 1, 1) * 0.1])
            
            masked_image[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = mask_value
        
        return masked_image
    
    def _saliency_based_mask(self, image_tensor, mask_ratio=0.3):
        """基于显著性的mask（简化版，用于训练时的快速处理）"""
        if self.mask_generator is None:
            return self._random_patch_mask(image_tensor)
            # 获取显著性图（简化计算以加速训练）
        print(image_tensor.grad.data)
        input_batch = image_tensor.unsqueeze(0).cuda()
        
        #gradients = input_batch.grad.data
        if torch.all(gradients == 0):
            print("梯度为零! 检查计算图连接")
            
        try:
            saliency_map = self.mask_generator.generate_attention_mask(
                input_batch, method='gradient'
                )
            #print("进行到这里了")    
                # 生成mask
            threshold = torch.quantile(saliency_map, 1 - mask_ratio)
            binary_mask = (saliency_map > threshold).float()
                
                # 应用mask
            binary_mask = binary_mask.cpu()
            masked_image = self.mask_generator.apply_mask_to_image(
                image_tensor.unsqueeze(0), binary_mask.unsqueeze(0))
            return masked_image.squeeze(0)
        except Exception as e:
                # 如果显著性计算失败，回退到随机遮挡
            print(f"Saliency mask failed, using random patches: {e}")
            print(f"Mask失败详情:\n{traceback.format_exc()}")
            return self._random_patch_mask(image_tensor)


class AdvancedMaskGenerator:
    """训练时使用的高效mask生成器 - 修复版本"""
    
    def __init__(self, clip_model, device='cuda'):
        self.clip_model = clip_model
        self.device = device
      
    def generate_gradient_mask(self, image_batch, target_logits, mask_ratio=0.3):
        """基于梯度的快速mask生成 - 修复版本"""
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


class MaskAwareContrastiveLoss(nn.Module):
    """支持mask的对比学习损失"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_orig, features_masked, labels):
        """
        计算原始特征和mask特征之间的对比损失
        Args:
            features_orig: 原始图像特征 [B, D]
            features_masked: 遮挡图像特征 [B, D]
            labels: 标签 [B]
        """
        batch_size = features_orig.shape[0]
        
        # 归一化特征
        features_orig = F.normalize(features_orig, dim=1)
        features_masked = F.normalize(features_masked, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_orig, features_masked.t()) / self.temperature
        
        # 创建正样本mask（相同标签）
        labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expand == labels_expand.t()).float()
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        loss = -torch.sum(positive_mask * log_prob) / torch.sum(positive_mask)
        
        return loss


def run_tip_adapter_F_with_mask(cfg, cache_keys, cache_values, val_features, val_labels, 
                                test_features, test_labels, clip_weights, clip_model, 
                                train_loader_F, mask_generator=None):
    """带mask增强的Tip-Adapter-F训练"""
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    
    # 初始化mask相关组件
    advanced_mask_gen = AdvancedMaskGenerator(clip_model) if mask_generator else None
    contrastive_loss = MaskAwareContrastiveLoss() if cfg.get('use_contrastive_loss', False) else None
    mask_generator = TipAdapterMaskGenerator(
            clip_model=clip_model,
            clip_weights=clip_weights,  # 从模型、配置文件或计算获取
            cache_keys=cache_keys,     # 通常是预提取的键向量张量
            cache_values=cache_values,  # 通常是预提取的值向量张量
            adapter_weights=adapter.weight, # 可能是适配器权重矩阵
            beta=beta,                  # 可能是标量超参数
            alpha=alpha                 # 可能是标量超参数
            )
    base_classifier = TipAdapterWrapper(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=adapter,
            beta=beta,
            alpha=alpha
        ).cuda()
    # 损失权重
    mask_loss_weight = cfg.get('mask_loss_weight', 0.1)
    contrastive_loss_weight = cfg.get('contrastive_loss_weight', 0.05)
    
    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        mask_loss_list = []
        
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            images = images.clone().detach().requires_grad_(True)
            #images=PGD_attack(base_classifier,images,target,epsilon=4/255,alpha=1/255,iters=100)
            #print(images.grad.data)
            batch_size = images.shape[0]
            
            # 编码原始图像
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 计算affinity和logits
            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            
            # 基础分类损失
            classification_loss = F.cross_entropy(tip_logits, target)
            
            total_loss = classification_loss
            print(f"Mask activated: {advanced_mask_gen is not None}, Prob: {cfg.get('mask_augment_prob', 0.3)}")
            if target.ndim == 1:
            # 如果是1D tensor，不需要指定dim
                numeric_labels = torch.argmax(target).float()
            elif target.ndim == 2:
            # 如果是2D tensor，使用dim=1
                numeric_labels = torch.argmax(target, dim=1).float()
            else:
            # 对于更高维度，需要指定正确的维度
                numeric_labels = torch.argmax(target, dim=-1).float()
            # 应用mask增强（如果启用）
            noise = torch.randn_like(images) * 0.12
            random_images=images+noise
            '''
            if advanced_mask_gen and random.random() < cfg.get('mask_augment_prob', 0.3):
                try:
                    # 生成mask
                    mask_ratio = random.uniform(0.2, 0.5)
                    
                    masks = advanced_mask_gen.generate_gradient_mask(
                        random_images, tip_logits, mask_ratio
                    )
                    #print(masks)
                    # 应用mask到图像
                    #masked_images = images * (1 - masks)

                    noise = torch.randn_like(images, device='cuda') * 0.12
                    noise = masks*noise
                    noise_m = torch.randn_like(images, device='cuda') * 0.25
                    noise_m = (1-masks)*noise_m
                    masked_images = images+noise+noise_m
                    #masked_images = images *masks
                    
                    # 编码mask图像
                    with torch.no_grad():
                        masked_features = clip_model.encode_image(masked_images)
                        masked_features /= masked_features.norm(dim=-1, keepdim=True)
                    
                    # 计算mask图像的logits
                    masked_affinity = adapter(masked_features)
                    masked_cache_logits = ((-1) * (beta - beta * masked_affinity)).exp() @ cache_values
                    masked_clip_logits = 100. * masked_features @ clip_weights
                    masked_tip_logits = masked_clip_logits + masked_cache_logits * alpha
                    
                    # Mask一致性损失：要求mask后的预测与原始预测一致
                    mask_consistency_loss = F.kl_div(
                        F.log_softmax(masked_tip_logits, dim=-1),
                        F.softmax(tip_logits.detach(), dim=-1),
                        reduction='batchmean'
                    )
                    
                    total_loss += mask_loss_weight * mask_consistency_loss
                    mask_loss_list.append(mask_consistency_loss.item())
                    
                    # 对比学习损失（可选）
                    if contrastive_loss:
                        contrast_loss = contrastive_loss(image_features, masked_features, target)
                        total_loss += contrastive_loss_weight * contrast_loss
                
                except Exception as e:
                    print(f"Mask augmentation failed: {e}")
                    # 继续使用基础损失
                    pass
            '''
            # 计算准确率
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(total_loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        avg_mask_loss = sum(mask_loss_list) / len(mask_loss_list) if mask_loss_list else 0.0
        
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}, Mask Loss: {:.4f}'.format(
            current_lr, correct_samples / all_samples, correct_samples, all_samples, 
            sum(loss_list)/len(loss_list), avg_mask_loss))
        
        # Eval
        adapter.eval()
        with torch.no_grad():
            affinity = adapter(test_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * test_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            acc = cls_acc(tip_logits, test_labels)
        
        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    return adapter, best_acc


def create_mask_augmented_loader(data_source, batch_size, transform, mask_generator=None, 
                                mask_augment_config=None):
    """创建支持mask增强的数据加载器"""
    
    if mask_augment_config is None:
        mask_augment_config = {
            'mask_prob': 0.3,
            'mask_strategies': ['random_patches', 'saliency_based'],
        }
    
    dataset = MaskAugmentedDataset(
        data_source=data_source,
        transform=transform,
        mask_generator=mask_generator,
        **mask_augment_config
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

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


class TipAdapterWrapper(torch.nn.Module):
    def __init__(self, clip_model, cache_keys, cache_values, clip_weights,adapter, beta=5.0, alpha=0.5):
        super().__init__()
        self.clip_model = clip_model
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.clip_weights = clip_weights
        self.adapter = adapter
        self.beta = beta
        self.alpha = alpha
       

       

    def forward(self, x):
        
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        
        # 模仿Tip-Adapter逻辑
        
        if self.adapter is not None:
            affinity = self.adapter(image_features)
        else:
            affinity = image_features @ self.cache_keys
      
        #affinity = features @ self.cache_keys
        cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
        clip_logits = 100.0 * image_features @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha
        
        return tip_logits
class TipAdapterWrapper_aom(torch.nn.Module):
    def __init__(self, clip_model, cache_keys, cache_values, clip_weights,adapter, beta=5.0, alpha=0.5):
        super().__init__()
        self.clip_model = clip_model
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.clip_weights = clip_weights
        self.adapter = adapter
        self.beta = beta
        self.alpha = alpha
       

       

    def forward(self, x_anchor,x_source):
        #image_features_x_anchor = self.clip_model.encode_image(x_anchor)
        #image_features_x_source = self.clip_model.encode_image(x_source)
        a=1.0
        image_features=a*x_anchor+(1-a)*x_source
        image_features=image_features / image_features.norm(dim=-1, keepdim=True)
        #image_features = self.clip_model.encode_image(x)
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        
        # 模仿Tip-Adapter逻辑
        
        if self.adapter is not None:
            affinity = self.adapter(image_features)
        else:
            affinity = image_features @ self.cache_keys
      
        #affinity = features @ self.cache_keys
        cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
        clip_logits = 100.0 * image_features @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha
        
        return tip_logits
import torch
import torch.nn.functional as F

class slerp(torch.nn.Module):
    """
    球面线性插值 (Slerp)
    :param v0: 起始特征向量 (f_anchor)
    :param v1: 目标特征向量 (f_adv)
    :param t: 插值系数 [0, 1]，0 返回 v0，1 返回 v1
    :param DOT_THRESHOLD: 阈值，当向量过于接近时切换为线性插值
    """
    def __init__(self, clip_model, cache_keys, cache_values, clip_weights,adapter, beta=5.0, alpha=0.5,few_shot=True):
        super().__init__()
        self.clip_model = clip_model
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.clip_weights = clip_weights
        self.adapter = adapter
        self.beta = beta
        self.alpha = alpha
        self.few_shot = True
    def forward(self,x_anchor,x_source):
    # 2. 计算夹角的余弦值
        dot = torch.sum(x_anchor * x_source, dim=-1, keepdim=True)
        t=-0.75

    # 3. 如果向量太接近，直接用线性插值以保证数值稳定性
        # 处理 batch 情况，避免 if 语句的歧义
        mask = torch.abs(dot) > 0.9995
        res_lerp = torch.lerp(x_anchor, x_source, t)
        # 线性插值结果

    # 4. 计算夹角 theta
        # 使用安全的 dot 值计算 Slerp，避免 acos 越界或除以零
        # mask 为 True 的地方用 0 替代（acos(0)=pi/2, sin(pi/2)=1），这些位置的结果最终会被 lerp 替代
        dot_safe = torch.where(mask, torch.zeros_like(dot), dot)
        
        theta_0 = torch.acos(dot_safe)
        theta_t = theta_0 * t

    # 5. 计算插值基底
        sin_theta_0 = torch.sin(theta_0)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        #s0 = torch.sin(theta_t -theta_0) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0

    # 6. 组合结果并恢复原始模长（可选）
    #更改了二者的比例
        res_slerp = s0 * x_anchor + s1 * x_source
        
        # 根据 mask 选择 Lerp 或 Slerp 的结果
        res = torch.where(mask, res_lerp, res_slerp)
        res = res/ res.norm(dim=-1, keepdim=True)
        if self.few_shot==False:
            tip_logits = 100. * res @ self.clip_weights
        else:
            if self.adapter is not None:
                affinity = self.adapter(res)
            else:
                affinity = res @ self.cache_keys
      
            cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
            clip_logits = 100.0 * res @ self.clip_weights
            tip_logits = clip_logits + cache_logits * self.alpha
        '''
        if self.adapter is not None:
            affinity = self.adapter(res)
        else:
            affinity = res @ self.cache_keys
      
        #affinity = features @ self.cache_keys
        cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
        clip_logits = 100.0 * res @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha
        '''
        return tip_logits

def evaluate_model_attack_rs(model_wrapper, model_wrapper_aom,test_loader, epsilon=4/255, alpha=1/255, iters=10,
                             num_samples=10, noise_std=0.12,mask_generator=None , clip_model=None,dataset_size=10):
    model_wrapper.eval()
    total_correct = 0
    total_samples = 0
    total_correct_aom=0
    total_samples_aom=0
    total_correct_ars=0
    total_correct_ars_aom=0
    our_aggregated_logits=0
    total_correct_our=0
    n0=100
    n=1000
    alpha1=0.001
    total_sigma=0.25
    first_query_budget_frac=torch.tensor(np.sqrt(0.25))
    advanced_mask_gen = AdvancedMaskGenerator(clip_model) if mask_generator else None
    
    for images, target in tqdm(test_loader, desc="Evaluating Attack with Random Smoothing"):
        images, target = images.to(device), target.to(device)
        images = images.clone().detach().requires_grad_(True)
        # 先对每个batch利用PGD产生对抗样本
        #adv_images = PGD_attack(model_wrapper, images, target, epsilon=1/255, alpha=1/255, iters=iters)
        atk=torchattacks.PGD(model_wrapper, eps=1/255, alpha=1/255, steps=10, random_start=True)
        #atk=torchattacks.AutoAttack(model_wrapper, norm='Linf', eps=1/255, version='standard', n_classes=dataset_size, seed=None, verbose=False)
        adv_images = atk(images,target)
        # 对每个对抗样本，进行num_samples次噪声采样，并平均输出
        batch_size = images.shape[0]
        aggregated_logits = torch.zeros(batch_size,dataset_size).to(device)  # val类别数为1000
        aggregated_logits_ars=torch.zeros(batch_size,dataset_size).to(device)
        image_feature=clip_model.encode_image(adv_images)
        images_transformed_feature=torch.zeros_like(image_feature).to(device)
        adv_images_feature=torch.zeros_like(image_feature).to(device)
        aom_feature=torch.zeros_like(image_feature).to(device)
        ars_feature=torch.zeros_like(image_feature).to(device)
        for _ in range(num_samples):
            sigma_1 = total_sigma / first_query_budget_frac
            noise = torch.randn_like(adv_images) * sigma_1
            first_noisy_images = adv_images + noise
            noise1=torch.randn_like(adv_images) * total_sigma
            noisy_images=adv_images+noise1
            noisy_images = torch.clamp(noisy_images, torch.min(noisy_images), torch.max(noisy_images))
            first_noisy_images = torch.clamp(first_noisy_images, torch.min(first_noisy_images), torch.max(first_noisy_images))
            if not first_noisy_images.requires_grad:
                first_noisy_images = first_noisy_images.clone().detach().requires_grad_(True)
            #print(f"first_noisy_images.requires_grad: {first_noisy_images.requires_grad}")
            logits = model_wrapper(first_noisy_images)
            mask_ratio = random.uniform(0.2, 0.5)
            masks = advanced_mask_gen.generate_gradient_mask(
                first_noisy_images, logits, mask_ratio
                )
            with torch.no_grad():
                #images_transformed = torch.mul(adv_images, masks)
                norm_2 = torch.sqrt(torch.tensor(3)) * torch.norm(masks.view(masks.shape[0], -1), p=2, dim=1)
                second_query_budget_frac = torch.sqrt(1 - torch.square(first_query_budget_frac))
                sigma_2 = (total_sigma /second_query_budget_frac) * (norm_2 / np.sqrt(3*224*224))
                sigma_2 = torch.maximum(torch.ones(sigma_2.shape).to(device) * 1e-6, sigma_2) 
                norm_dist = D.Normal(loc=0., scale=sigma_2)
                #noise = torch.randn_like(images, device='cuda') * sigma_2
                sigma_2 = sigma_2.view(-1, 1, 1, 1)
                noise = torch.randn_like(images, device='cuda') * sigma_2.expand_as(images)
                images_transformed = torch.mul(adv_images+noise, masks)
                #images_transformed += noise
                pre_averaging_x_transformed = images_transformed.clone().detach()
                sigma_1 = sigma_1.repeat(sigma_2.shape[0]).reshape(sigma_2.shape[0],1,1,1).repeat(1,1,224,224).to(device)
                sigma_2 = sigma_2.reshape(sigma_2.shape[0],1,1,1).repeat(1,1,224,224).to(device)
                denominator = ((masks ** 2) * (sigma_1 ** 2)) + (sigma_2 ** 2).to(device)
                w1 = sigma_2 ** 2 / denominator
                w2 = ((sigma_1 ** 2) * (masks)) / denominator
                images_transformed *= w2
                #images_transformed_2=copy.deepcopy(images_transformed)
                #images_transformed_2=images_transformed.clone().detach().requires_grad_(True)
                images_transformed_2=pre_averaging_x_transformed*w2
                images_transformed += (w1 * adv_images)
                images_transformed_2+=(w1 * first_noisy_images)
                images_transformed_feature+=clip_model.encode_image(images_transformed)
                adv_images_feature+=clip_model.encode_image(adv_images)
                aom_feature+=clip_model.encode_image(noisy_images)
                ars_feature+=clip_model.encode_image(images_transformed_2)
                logits = model_wrapper(images_transformed_2)
                #our_logits = model_wrapper(images_transformed)
            aggregated_logits_ars += logits
            #our_aggregated_logits+=our_logits
        images_transformed_feature/=num_samples
        adv_images_feature/=num_samples
        aom_feature/=num_samples
        aggregated_logits_ars /= num_samples
        ars_feature/=num_samples
        #our_aggregated_logits/=num_samples
        aggregated_logits=model_wrapper_aom(images_transformed_feature,adv_images_feature)
        aom_logits=model_wrapper_aom(aom_feature,adv_images_feature)
        ars_aom_logits=model_wrapper_aom(ars_feature,adv_images_feature)
        preds = aggregated_logits.argmax(dim=1)
        preds_aom=aom_logits.argmax(dim=1)
        preds_ars=aggregated_logits_ars.argmax(dim=1)
        preds_ars_aom=ars_aom_logits.argmax(dim=1)
        #preds_our=our_aggregated_logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)
        total_correct_aom += (preds_aom == target).sum().item()
        total_correct_ars+=(preds_ars == target).sum().item()
        total_correct_ars_aom+=(preds_ars_aom == target).sum().item()
        #total_correct_our+=(preds_our == target).sum().item()
        total_samples_aom += target.size(0)
    acc = 100. * total_correct / total_samples
    aom_acc=100.*total_correct_aom/total_samples_aom
    ars_acc=100.*total_correct_ars/total_samples_aom
    ars_aom_acc=100.*total_correct_ars_aom/total_samples_aom
    #our_acc=100.*total_correct_our/total_samples_aom
    return acc,aom_acc,ars_acc,ars_aom_acc
def evaluate_model_rs(model_wrapper, test_loader, epsilon=4/255, alpha=1/255, iters=10,
                             num_samples=10, noise_std=0.12,num_classes=100):
    model_wrapper.eval()
    total_correct = 0
    total_samples = 0
    smoothed_classifier = Smooth(
            base_classifier=model_wrapper,
            num_classes=num_classes,
            sigma=noise_std
        )
    n0=100
    n=1000
    alpha1=0.001
    cer_radius= []
    for images, target in tqdm(test_loader, desc="Evaluating Attack with Random Smoothing"):
        images, target = images.cuda(),target.cuda()
        # 先对每个batch利用PGD产生对抗样本
        adv_images = PGD_attack(model_wrapper, images, target, epsilon=epsilon, alpha=alpha, iters=iters)
        # 对每个对抗样本，进行num_samples次噪声采样，并平均输
        prediction, radius = smoothed_classifier.certify(
                x=images,
                n0=n0,
                n=n,
                alpha=alpha1,
                batch_size=64
            )
        cer_radius.append(radius)
        if prediction == target.item():
            total_correct +=1
        #total_correct += (prediction == target.item()).sum().item()
        total_samples += target.size(0)
    acc = 100. * total_correct / total_samples
    return acc,np.mean(cer_radius)
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
    cfg = yaml.load(open("./configs/eurosat.yaml", 'r'), Loader=yaml.Loader)
    few_shot=True
    # 添加mask增强相关配置
    mask_config = {
        'use_mask_augment': True,
        'mask_augment_prob': 0.5,  # 使用mask增强的概率
        'mask_loss_weight': 0.1,   # mask一致性损失权重
        'use_contrastive_loss': False,
        'contrastive_loss_weight': 0.05,
        'mask_strategies': ['random_patches', 'saliency_based'],
        'mask_prob': 0.5,  # 数据集中应用mask的概率
        'shots':4

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
    # 初始化mask生成器（用于训练）
    mask_generator = None
    if cfg.get('use_mask_augment', True):
        try:
            from mask_generator_module import TipAdapterMaskGenerator
            # 这里需要根据实际情况初始化mask生成器
            #mask_generator = TipAdapterMaskGenerator(...)
            mask_generator = TipAdapterMaskGenerator(
            clip_model=clip_model,
            clip_weights=clip_weights,  # 从模型、配置文件或计算获取
            cache_keys=cache_keys,     # 通常是预提取的键向量张量
            cache_values=cache_values,  # 通常是预提取的值向量张量
            adapter_weights=adapter.weight, # 可能是适配器权重矩阵
            beta=1,                  # 可能是标量超参数
            alpha=1.7                 # 可能是标量超参数
            )
            print("Mask generator initialized for training augmentation.")
        except ImportError:
            print("Mask generator not available, using random masking only.")
    
    # 创建带mask增强的训练数据加载器
    if cfg.get('use_mask_augment', False):
        mask_augment_config = {
            'mask_prob': cfg['mask_prob'],
            'mask_strategies': cfg['mask_strategies']
        }
        print("使用了这个")
        '''
        train_loader_F = create_mask_augmented_loader(
            data_source=dataset.train_x,
            batch_size=256,
            transform=train_transform,
            mask_generator=mask_generator,
            mask_augment_config=mask_augment_config
        )
        '''
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=True)
    else:
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, 
                                          tfm=train_transform, is_train=True, shuffle=True)
    if few_shot:
        # 运行带mask增强的训练
        adapter, best_acc = run_tip_adapter_F_with_mask(
        cfg, cache_keys, cache_values, val_features, val_labels, 
        test_features, test_labels, clip_weights, clip_model, 
        train_loader_F, mask_generator
        )
    
        print(f"\n**** Final best accuracy with mask augmentation: {best_acc:.2f}% ****")
    
    # 超参数搜索
        print("\n-------- Searching hyperparameters on the val set. --------")
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, 
                                     val_labels, clip_weights, adapter=adapter)
    
        print("\n-------- Final evaluation on the test set. --------")
        with torch.no_grad():
            affinity = adapter(test_features)
            cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
            clip_logits = 100. * test_features @ clip_weights
            tip_logits = clip_logits + cache_logits * best_alpha
            final_acc = cls_acc(tip_logits, test_labels)
        smooth_adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
        smooth_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
        print("**** Final Tip-Adapter-F test accuracy with mask augmentation: {:.2f}%. ****".format(final_acc))
        base_classifier = TipAdapterWrapper(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=smooth_adapter,
            beta=best_beta,
            alpha=best_alpha
        ).cuda()
        model_wrapper_aom=TipAdapterWrapper_aom(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=smooth_adapter,
            beta=best_beta,
            alpha=best_alpha,
            few_shot=few_shot
        ).cuda()
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    model_wrapper_aom_zero=slerp(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=adapter,
            beta=cfg['init_beta'],
            alpha=cfg['init_beta'],
            few_shot=few_shot
        ).cuda()
    attack_epsilon=1/255
    attack_alpha = 1/255  # 可根据需求调整
    attack_iters = 10
    num_class=dataset.num_classes
    #adv_acc=evaluate_model_attack(base_classifier, test_loader, epsilon= 1/255, alpha=1/255, iters=100)
    #print("adv_acc:{:.2f}%".format(adv_acc))
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    zero_shot_net = ZeroShotClassifier(clip_model, clip_weights).to(device)
    acc_clip,aom_acc,ars_acc,ars_aom_acc = evaluate_model_attack_rs(base_classifier,model_wrapper_aom, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=5, noise_std=0.25,mask_generator=mask_generator,clip_model=clip_model,dataset_size=num_class)
    print("Zero-shot CLIP Accuracy under PGD attack: {:.2f}%".format(acc_clip))
    print("aom Accuracy under PGD attack: {:.2f}%".format(aom_acc))
    print("ars Accuracy under PGD attack: {:.2f}%".format(ars_acc))
    print("ars_aom Accuracy under PGD attack: {:.2f}%".format(ars_aom_acc))
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