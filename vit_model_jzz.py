"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torchvision.models import resnet50


# 多级特征适配器模块
class MultiStageAdapter(nn.Module):
    def __init__(self, cnn_channels=[256, 512, 1024], vit_dim=768, fusion_dim=256):
        super().__init__()
        
        # 为每个CNN阶段定义适配器
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, vit_dim, kernel_size=1),  # 通道对齐
                nn.BatchNorm2d(vit_dim),
                nn.GELU(),
                nn.Conv2d(vit_dim, vit_dim, 3, padding=1)  # 空间信息保持
            ) for ch in cnn_channels
        ])
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(vit_dim*len(cnn_channels), fusion_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, vit_dim, 3, padding=1)
        )
        
        # 上采样器（用于对齐不同尺度的特征）
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
            for i in range(len(cnn_channels)-1, 0, -1)
        ])

    def forward(self, features):
        """
        features: list of CNN特征，从浅到深排序
                  [stage1_feat, stage2_feat, stage3_feat]
        """
        # 处理每个适配器
        adapted_feats = []
        for feat, adapter in zip(features, self.adapters):
            x = adapter(feat)
            adapted_feats.append(x)
        
        # 上采样对齐尺寸（以最深层特征为基准）
        target_size = adapted_feats[-1].shape[-2:]
        for i in range(len(adapted_feats)-1):
            adapted_feats[i] = self.upsamplers[i](adapted_feats[i])
            adapted_feats[i] = torch.nn.functional.interpolate(
                adapted_feats[i], size=target_size, mode='bilinear')
        
        # 通道维度拼接
        fused = torch.cat(adapted_feats, dim=1)  # [B, vit_dim*3, H, W]
        
        # 融合后处理
        return self.fusion(fused)  # [B, vit_dim, H, W]





def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    # 图像的通道数：3 嵌入维度：768 窗口大小：16 是否使用层归一化：None
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 224 / 16 = 14 => 14 * 14 = 196张图，每张图的尺寸为 16 * 16 * 3 = 768
        # grid_size = ( 14 , 14 )
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 14 * 14 = 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # embed_dim = 768 kernel_size = 16 stride = 16
        # 3 -> 768
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # (B * 3 * 224 * 224) -> (B * 768 * 14 * 14) -> (B * 768 * 196 ) -> (B * 196 * 768)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim 为768
                 num_heads=8,
                 qkv_bias=False, # 是否在查询（Q）、键（K）和值（V）变换中使用偏置
                 qk_scale=None, #  缩放因子
                 attn_drop_ratio=0., # 注意力权重后的 dropout 比率
                 proj_drop_ratio=0.): # 最终投影后的 dropout 比率
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # 这里qkv共用了一个linear层，因为输出的通道为dim * 3，所以可以等同三个qkv矩阵（理解为三个矩阵拼在一起了），这样的设计或许利于并行化
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] : [B ,197 ,768]
        B, N, C = x.shape

        # qkv():[batch_size, num_patches + 1, total_embed_dim] -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head] 为了好计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #  如果启用了蒸馏（distilled=True），则有两个特殊 tokens (cls_token 和 dist_token)；否则只有一个 (cls_token)
        self.num_tokens = 2 if distilled else 1
        # 如果用户没有提供特定的归一化层，则使用 LayerNorm（带有 eps=1e-6）作为默认选项。
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 如果用户没有指定激活函数，则使用 GELU 作为默认激活函数。
        act_layer = act_layer or nn.GELU


        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 创建一个可训练的分类 token (cls_token)，其形状为 [1, 1, embed_dim]，并且所有元素初始化为零
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))# [1,197,768]
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 生成一个递增的丢弃概率列表，用于实现随机深度（stochastic depth）。这有助于提高泛化能力，特别是在深度较大的网络中
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # 如果设置了 representation_size 并且不是蒸馏版本，则创建一个额外的线性层和激活函数（Tanh），用于将特征映射到指定大小的空间
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed) # [B, 197, 768] 广播机制
        x = self.blocks(x)
        x = self.norm(x)
        # print(x.shape)#torch.Size([B, 197, 768])
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)#torch.Size([B, 768])
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            print(x.shape) # torch.Size([B, 21843])
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes=21843, has_logits=True):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes
    )
    return model

def vit_base_patch16_224_in21k_2(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/downloadv0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth/
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              drop_ratio=0.1,         # 增加输入dropout
                              attn_drop_ratio=0.1,    # 增加注意力dropout
                              drop_path_ratio=0.2,    # 增加随机深度
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model

# 多级特征ViT主模型
class MultiScaleFusionViT(nn.Module):
    def __init__(self, num_classes=1000, vit_model=None):
        super().__init__()
        
        # 1. CNN特征提取器（ResNet50前三个stage）
        resnet = resnet50(pretrained=True)
        self.cnn_stages = nn.ModuleDict({
            'stem': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            'stage1': resnet.layer1,  # stride 4
            'stage2': resnet.layer2,  # stride 8
            'stage3': resnet.layer3   # stride 16
        })
        
        # 2. 多级特征适配器
        self.adapter = MultiStageAdapter(
            cnn_channels=[256, 512, 1024],  # ResNet各stage输出通道
            vit_dim=768,
            fusion_dim=512
        )
        
        # 3. 冻结的ViT骨干
        if vit_model is None:
            self.vit = VisionTransformer(
                img_size=14,  # 适配器输出特征图尺寸
                patch_size=1,  # 每个"patch"对应1x1的特征
                in_chans=768,
                num_classes=num_classes
            )
        else:
            self.vit = vit_model
            
         # 4. CLS Token和位置编码（保持与ViT兼容）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))  # 14x14=196 +1
        
        self._freeze_vit()
        
    def _freeze_vit(self):
        # 冻结ViT所有参数
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 可选：解冻LayerNorm和位置编码
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
        self.pos_embed.requires_grad = True

    def forward(self, x):
        # 步骤1：提取多级CNN特征
        cnn_features = []
        x = self.cnn_stages['stem'](x)
        x = self.cnn_stages['stage1'](x)  # [B,256,56,56]
        cnn_features.append(x)
        x = self.cnn_stages['stage2'](x)  # [B,512,28,28]
        cnn_features.append(x)
        x = self.cnn_stages['stage3'](x)  # [B,1024,14,14]
        cnn_features.append(x)
        
        # 步骤2：多级特征融合
        vit_feat = self.adapter(cnn_features)  # [B,768,14,14]
        
        # 步骤3：准备ViT输入
        B, C, H, W = vit_feat.shape
        vit_feat = vit_feat.flatten(2).transpose(1, 2)  # [B,196,768]
        
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        vit_feat = torch.cat([cls_tokens, vit_feat], dim=1)  # [B,197,768]
        
        # 添加位置编码
        vit_feat += self.pos_embed
        
        # 步骤4：通过冻结的ViT
        vit_feat = self.vit.blocks(vit_feat)
        vit_feat = self.vit.norm(vit_feat)
        
        # 分类头
        return self.vit.head(vit_feat[:, 0])


# 使用示例
if __name__ == '__main__':
    
    # 创建多级融合模型
    model = vit_base_patch16_224_in21k(num_classes=1000)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # 测试前向
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")  # [2,1000]
    
    
    
