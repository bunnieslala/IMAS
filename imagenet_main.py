import os
import time
import random
import argparse
import yaml
from tqdm import tqdm
import torchattacks
import torch
import torch.nn.functional as F
import torch.nn as nn
import core
from datasets.imagenet import ImageNet
import torchvision.transforms as transforms
from datasets.utils import build_data_loader
import clip
from utils import *
from core import Smooth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from mask_generator_module import TipAdapterMaskGenerator
import multiprocessing
import torch.distributions as D
def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

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
                    print("使用的是随机mask")
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
def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    sigma=0.12

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                noise = torch.randn_like(images, device='cuda') * sigma
                image_features = clip_model.encode_image(images+noise)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

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

    # Search Hyperparameters
    best_beta,best_alpha = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)
    data = struct.pack('ff', best_beta, best_alpha)
    with open('data.bin', 'wb') as f:
        f.write(data)


def run_tip_adapter_F_with_mask(cfg, cache_keys, cache_values, 
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
    
    # 损失权重
    mask_loss_weight = cfg.get('mask_loss_weight', 0.1)
    contrastive_loss_weight = cfg.get('contrastive_loss_weight', 0.05)
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



def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item())
    acc = 100 * acc / target.shape[0]
    return acc


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

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights



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


class TipAdapterClassifier(nn.Module):
    def __init__(self, clip_model, clip_weights, cache_keys, cache_values, beta, alpha):
        super(TipAdapterClassifier, self).__init__()
        self.clip_model = clip_model
        self.clip_weights = clip_weights
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.beta = beta
        self.alpha = alpha

    def forward(self, image):
        features = self.clip_model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        affinity = features @ self.cache_keys
        # 计算cache_logits：注意这里使用exp(-1*(beta - beta*affinity))
        cache_logits = torch.exp(-1 * (self.beta - self.beta * affinity)) @ self.cache_values
        clip_logits = 100. * features @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha
        return tip_logits


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
       
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
       

    def forward(self, x):
        # 归一化输入
        #x = self.normalize(x)
        # 提取图像特征
        
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Tip-Adapter逻辑
        
        if self.adapter is not None:
            affinity = self.adapter(image_features)
        else:
            affinity = image_features @ self.cache_keys
      
        #affinity = features @ self.cache_keys
        cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
        clip_logits = 100.0 * image_features @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha
        
        return tip_logits



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
                x=adv_images,
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
        a=1.25
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
class slerp(torch.nn.Module):
    def __init__(self, clip_model, cache_keys, cache_values, clip_weights,adapter, beta=5.0, alpha=0.5):
        super().__init__()
        self.clip_model = clip_model
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.clip_weights = clip_weights
        self.adapter = adapter
        self.beta = beta
        self.alpha = alpha
    def forward(self,x_anchor,x_noise,x_source):
    # 2. 计算夹角的余弦值
        dot = torch.sum(x_anchor * x_source, dim=-1, keepdim=True)
        dot1= torch.sum(x_noise * x_source, dim=-1, keepdim=True)
        #t = torch.where(dot1 > dot, torch.tensor(0.75, device=dot.device), torch.tensor(-0.75, device=dot.device))
        t = torch.where(dot1 > dot, -0.75, 0.75).to(dot.device).to(dot.dtype)

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
        if self.adapter is not None:
            affinity = self.adapter(res)
        else:
            affinity = res @ self.cache_keys
      
        #affinity = features @ self.cache_keys
        cache_logits = torch.exp(-self.beta * (1 - affinity)) @ self.cache_values
        clip_logits = 100.0 * res @ self.clip_weights
        tip_logits = clip_logits + cache_logits * self.alpha

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
        atk=torchattacks.PGD(model_wrapper, eps=1/255, alpha=1/255, steps=10, random_start=True)
        #atk=torchattacks.AutoAttack(model_wrapper, norm='Linf', eps=1/255, version='standard', n_classes=dataset_size, seed=None, verbose=False)
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
def main():

    cfg = yaml.load(open("./configs/imagenet.yaml", 'r'), Loader=yaml.Loader)
    mask_config = {
        'use_mask_augment': True,
        'mask_augment_prob': 0.5,  # 使用mask增强的概率
        'mask_loss_weight': 0.1,   # mask一致性损失权重
        'use_contrastive_loss': False,
        'contrastive_loss_weight': 0.05,
        'mask_strategies': ['random_patches', 'saliency_based'],
        'mask_prob': 0.5,  # 数据集中应用mask的概率
        'shots':16
    }
    cfg.update(mask_config)
   
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    #test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    test_features, test_labels = pre_load_features(cfg,"test",cache_keys, cache_values, clip_model, test_loader, clip_weights)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    #run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    #run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)
    '''
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
    '''
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    mask_generator = TipAdapterMaskGenerator(
            clip_model=clip_model,
            clip_weights=clip_weights,  # 从模型、配置文件或计算获取
            cache_keys=cache_keys,     # 通常是预提取的键向量张量
            cache_values=cache_values,  # 通常是预提取的值向量张量
            adapter_weights=adapter.weight, # 可能是适配器权重矩阵
            beta=1,                  # 可能是标量超参数
            alpha=1.7                 # 可能是标量超参数
            )
    
    '''
    adapter, best_acc = run_tip_adapter_F_with_mask(
        cfg, cache_keys, cache_values,
        test_features, test_labels, clip_weights, clip_model, 
        train_loader_F, mask_generator
    )
    '''
    #adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    #adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    affinity = adapter(test_features)
    best_beta,best_alpha = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)
    
    zero_shot_net = ZeroShotClassifier(clip_model, clip_weights).to(device)
    tip_adapter_net = TipAdapterClassifier(clip_model, clip_weights,cache_keys, cache_values, best_beta,best_alpha).to(device)
    base_classifier = TipAdapterWrapper(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=adapter,
            beta=best_beta,
            alpha=best_alpha
        ).cuda()
    
    
    #ai随机平滑代码
    use_smoothing = False  # 设置为True启用随机平滑
    sigma = 0.12        # 噪声水平
    n0 = 100              # 选择样本数
    n = 1000             # 估计样本数
    alpha1 = 0.001          # 置信水平
    batch_size = 16       # 批处理大小
    #beta, alpha = run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)
    #smooth_adapter= nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    #smooth_adapter.weight =torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    #smooth_adapter.eval()
    # print("\n========= Begin PGD Attack Evaluation (iters=10) =========\n")
    
    attack_epsilon = 1/255  # 可根据需求调整
    attack_alpha = 1/255  # 可根据需求调整
    attack_iters = 100
    #adv_acc=evaluate_model_attack(base_classifier, test_loader, epsilon= 1/255, alpha=1/255, iters=100)
    #print("adv_acc:{:.2f}%".format(adv_acc))
    model_wrapper_aom=TipAdapterWrapper_aom(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            adapter=adapter,
            beta=1,
            alpha=1.7
        ).cuda()
    zero_shot_net = ZeroShotClassifier(clip_model, clip_weights).to(device)
    print("Evaluating Zero-shot CLIP under PGD Attack ...")
    acc_clip,aom_acc,ars_acc,ars_aom_acc,clean_acc = evaluate_model_attack_rs(zero_shot_net,model_wrapper_aom, test_loader, epsilon=attack_epsilon, alpha=attack_alpha, iters=attack_iters,
    num_samples=10, noise_std=0.25,clip_model=clip_model,dataset_size=1000)
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
    '''
    print("Evaluating Zero-shot CLIP under auto Attack ...")
    acc_clip = evaluate_model_attack_rs(base_classifier, test_loader, epsilon=4/255, alpha=1/255, iters=10,
                             num_samples=10, noise_std=0.25)
    print("Zero-shot CLIP Accuracy under auto attack: {:.2f}%".format(acc_clip))
    #
    print("Evaluating Zero-shot CLIP under auto Attack ...")
    acc_clip = evaluate_model_attack_rs(base_classifier, test_loader, epsilon=4/255, alpha=1/255, iters=10,
                             num_samples=10, noise_std=0.5)
    print("Zero-shot CLIP Accuracy under auto attack: {:.2f}%".format(acc_clip))
    print("Evaluating Zero-shot CLIP under auto Attack ...")
    
    print("Evaluating tip-adapter CLIP under PGD Attack ...")
    acc_tip_clip = evaluate_model_attack(base_classifier, test_loader, epsilon=attack_epsilon, alpha=attack_alpha,
    iters=attack_iters)
    print("tip-adapter CLIP Accuracy under PGD attack: {:.2f}%".format(acc_tip_clip))
    '''

    '''
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

    if use_smoothing:
        print("\n-------- Randomized Smoothing Certification --------")
        
        # 创建封装模型
        base_classifier = TipAdapterClassifier(
            clip_model=clip_model,
            cache_keys=cache_keys,
            cache_values=cache_values,
            clip_weights=clip_weights,
            beta=best_beta,
            alpha=best_alpha
        ).cuda()
        
        # 初始化平滑分类器
        smoothed_classifier = Smooth(
            base_classifier=base_classifier,
            num_classes=clip_weights.size(0),
            sigma=sigma
        )

        # 创建原始图像测试集（无归一化）
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        test_loader_raw = build_data_loader(
            data_source=imagenet.test,
            batch_size=1,
            is_train=False,
            tfm=test_transform,
            shuffle=False
        )


        # 认证测试
        certified_radii = []
        for i, (image, target) in enumerate(tqdm(test_loader)):
            image, target = image.cuda(), target.cuda()
            #image,label = test_loader.dataset[i]
            #image =image.cuda()
            
            # 使用随机平滑进行认证
            prediction, radius = smoothed_classifier.certify(
                x=image,
                n0=n0,
                n=n,
                alpha=alpha1,
                batch_size=batch_size
            )
            print(target)
            if prediction ==target.item():
                certified_radii.append(radius)
            else:
                certified_radii.append(0.0)
        
        # 输出统计结果
        clean_acc = np.mean([r > 0 for r in certified_radii])
        mean_radius = np.mean(certified_radii)
        print(f"Certified Accuracy: {clean_acc:.2%}")
        print(f"Mean Certified Radius: {mean_radius:.4f}")  

if __name__ == '__main__':
    main()