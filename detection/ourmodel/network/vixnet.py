import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import os
from torchvision.models.efficientnet import EfficientNet_B4_Weights
import timm
from timm.models.efficientnet import tf_efficientnetv2_s

model_urls = {
    'xception':'<url id="d0eulfd3v89ulq7l6fe0" type="url" status="failed" title="" wc="0">https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1</url> '
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1):
        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


# ViT相关模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.wq = nn.Linear(key_dim, key_dim)
        self.wk = nn.Linear(key_dim, key_dim)
        self.wv = nn.Linear(key_dim, key_dim)
        self.sqrt_key_dim = torch.sqrt(torch.FloatTensor([key_dim]))

    def forward(self, query, key, value):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # 确保所有张量都在同一个设备上
        device = q.device
        self.sqrt_key_dim = self.sqrt_key_dim.to(device)

        q = q.view(-1, self.num_heads, self.key_dim)
        k = k.view(-1, self.num_heads, self.key_dim)
        v = v.view(-1, self.num_heads, self.key_dim)

        scores = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_key_dim
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.view(-1, self.key_dim)
        return output


# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads, key_dim):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.key_dim = key_dim
#         self.head_dim = key_dim // num_heads
        
#         assert self.head_dim * num_heads == key_dim, "key_dim必须能被num_heads整除"
        
#         self.wq = nn.Linear(key_dim, key_dim)
#         self.wk = nn.Linear(key_dim, key_dim)
#         self.wv = nn.Linear(key_dim, key_dim)
        
#     def forward(self, query, key, value):
#         # 输入形状: [B, N, C] (N是序列长度，这里N=1)
#         B, N, _ = query.shape
        
#         q = self.wq(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, N, hd]
#         k = self.wk(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # [B, nh, N, hd]
#         v = self.wv(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, N, hd]
        
#         # 计算注意力分数
#         attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
#         attn = F.softmax(attn, dim=-1)
        
#         # 应用注意力权重
#         output = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.key_dim)
#         return output
    

class LayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

# class ViTBlock(nn.Module):
#     def __init__(self, num_heads=8, dim=768):
#         super(ViTBlock, self).__init__()
#         self.ln1 = nn.LayerNorm(dim)  # 使用内置LayerNorm
#         self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim)
#         self.ln2 = nn.LayerNorm(dim)  # 使用内置LayerNorm
#         self.dense1 = nn.Linear(dim, dim * 4)
#         self.gelu = nn.GELU()
#         self.dense2 = nn.Linear(dim * 4, dim)

#     def forward(self, x):
#         # 保持输入输出形状一致 [B, N, C]
#         # Self-attention
#         attn = self.mha(self.ln1(x), self.ln1(x), self.ln1(x))
#         x = x + attn
        
#         # Feed-forward
#         x_dense = self.dense2(self.gelu(self.dense1(self.ln2(x))))
#         x = x + x_dense
#         return x

class ViTBlock(nn.Module):
    def __init__(self, num_heads=8, dim=768):
        super(ViTBlock, self).__init__()
        self.ln1 = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.add1 = nn.Identity()
        self.ln2 = LayerNormalization()
        self.dense1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(dim * 4, dim)
        self.add2 = nn.Identity()

    def forward(self, x):
        # Self-attention
        x_norm = self.ln1(x)
        attn = self.mha(x_norm, x_norm, x_norm)
        x = self.add1(attn + x)

        # Feed-forward
        x_norm = self.ln2(x)
        x_dense = self.dense1(x_norm)
        x_dense = self.gelu(x_dense)
        x_dense = self.dense2(x_dense)
        x = self.add2(x_dense + x)

        return x



# class XceptionWithViT(nn.Module):
#     def __init__(self, num_classes=1):
#         super(XceptionWithViT, self).__init__()
#         self.xception = Xception(num_classes=1)
#         self.xception.fc = nn.Identity()  # 禁用原始分类器
        
#         self.proj = nn.Linear(2048, 768)
#         self.vit_blocks = nn.Sequential(
#             ViTBlock(num_heads=8, dim=768),
#             ViTBlock(num_heads=8, dim=768),
#             ViTBlock(num_heads=8, dim=768),
#             ViTBlock(num_heads=8, dim=768)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(768, num_classes),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.xception.features(x)      # [B, 2048, 1, 1]
#         x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [B, 2048]
        
#         x = self.proj(x).unsqueeze(1)      # [B, 1, 768]
#         x = self.vit_blocks(x)             # [B, 1, 768]
        
#         x = x.squeeze(1)                   # [B, 768]
#         return self.classifier(x)          # [B, 1]
    
class XceptionWithViT(nn.Module):
    def __init__(self, num_classes=1):
        super(XceptionWithViT, self).__init__()
        self.xception = Xception(num_classes=1000)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.reshape = nn.Flatten()
        self.dense_proj = nn.Linear(2048, 768)

        self.vit_blocks = nn.Sequential(
            ViTBlock(num_heads=8, dim=768),
            ViTBlock(num_heads=8, dim=768),
            ViTBlock(num_heads=8, dim=768),
            ViTBlock(num_heads=8, dim=768)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.xception.features(x)
        x = self.global_avg_pool(x)
        x = self.reshape(x)
        x = self.dense_proj(x)

        x = x.unsqueeze(1)  # 添加序列维度
        x = self.vit_blocks(x)
        x = x.squeeze(1)  # 移除序列维度

        x = self.classifier(x)
        return x


class EnhancedDeepFakeDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # 主干网络
        self.backbone = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # 修改分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # 移除原始分类器
        
        # 增强特征处理
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, 1792),
            nn.BatchNorm1d(1792),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(1792, 896),
            nn.BatchNorm1d(896),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(896, num_classes),
            nn.Sigmoid()
        )
        
        # 初始化权重
        if not pretrained:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.backbone(x)
        # 特征增强
        x = self.feature_processor(x)
        # 分类
        return self.classifier(x)

class EfficientNetV2WithClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(EfficientNetV2WithClassifier, self).__init__()
        try:
            self.efficientnet_v2 = timm.create_model('efficientnetv2_s', pretrained=pretrained)
        except RuntimeError as e:
            print(f"Error creating model with pretrained={pretrained}: {e}")
            print("Using pretrained=False instead.")
            self.efficientnet_v2 = timm.create_model('efficientnetv2_s', pretrained=False)
        
        # 修改最后的分类头
        in_features = self.efficientnet_v2.classifier.in_features
        self.efficientnet_v2.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnet_v2(x)
    

def effiv2(pretrained=False, **kwargs):
    model = EfficientNetV2WithClassifier(pretrained=pretrained, **kwargs)
    if pretrained:
        # Load pre-trained weights if needed
        # 这里可以根据需要实现加载预训练权重的逻辑
        # 例如：
        # model.load_state_dict(torch.load(pretrained_weight_path))
        pass
    return model

def efficientnet(pretrained=False, **kwargs):
    """
    构建增强型Deepfake检测模型（兼容原接口）
    参数：
        pretrained - 是否使用ImageNet预训练权重
        **kwargs - 兼容参数（实际使用num_classes参数）
    """
    return EnhancedDeepFakeDetector(
        num_classes=kwargs.get('num_classes', 1),
        pretrained=pretrained
    )

# 修改后的xception函数
def vitxception(pretrained=False, **kwargs):
    """
    Construct Xception model, optionally with ViT integration.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: Additional keyword arguments
    """
    model = XceptionWithViT(**kwargs) 
    if pretrained:
        # Load pre-trained weights if available
        # 这里需要实现加载预训练权重的逻辑
        # 例如：
        # model.load_state_dict(model_zoo.load_url(model_urls['xception']))
        pass
    return model