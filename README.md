# ResNet-ViT
基于Adapter构建ResNet-ViT的模型融合架构，提升ViT在小数据集上的性能，小参数量学习：仅训练Adapter和分类头
支持vit_base_patch16_224_in21k，vit_base_patch32_224，vit_base_patch32_224_in21k，vit_large_patch16_224，vit_large_patch16_224_in21k，vit_large_patch32_224_in21k，vit_large_patch32_224_in21k的ViT预训练模型，代码位于master分支
# 数据集结构
dataset  
  --种类1  
  --种类2  
  --种类3等等
# 用融合模型训练数据集
python train.jzz --data-path ./dataset1 --model-type fusion --weights ./vit_base_patch16_224_1K.pth --model-name vit_base_patch16_224 --freeze-layers
# 用原始ViT模型训练数据集
python train.jzz --data-path ./dataset1 --model-type original --weights ./vit_base_patch16_224_1K.pth --model-name vit_base_patch16_224
