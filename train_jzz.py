import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model_jzz import *
from utils_Copy1 import read_split_data, train_one_epoch, evaluate, plot_loss_acc

def create_model(model_name, num_classes=None, has_logits=None):
    # 获取当前模块的全局函数字典
    model_func = globals().get(model_name)
    
    if not model_func:
        raise ValueError(f"Model {model_name} not found")
    
    # 根据函数参数决定如何调用
    if "in21k" in model_name:
        if has_logits is not None:
            return model_func(num_classes=num_classes, has_logits=has_logits)
        return model_func(num_classes=num_classes)
    else:
        if num_classes is not None:
            return model_func(num_classes=num_classes)
        return model_func()


# 在main函数中修改模型初始化部分
def main(args):
    # ... 前面的代码保持不变 ...
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建 weights 文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 创建 runs/exp 文件夹
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    # 找到最新的 exp 文件夹编号
    exp_folders = [f for f in os.listdir(runs_dir) if f.startswith("exp")]
    if exp_folders:
        latest_exp = max(exp_folders, key=lambda x: int(x[3:]))
        new_exp_num = int(latest_exp[3:]) + 1
    else:
        new_exp_num = 1

    # 创建新的 exp 文件夹
    save_dir = os.path.join(runs_dir, f"exp{new_exp_num}")
    os.makedirs(save_dir, exist_ok=True)

    # 初始化 TensorBoard
    tb_writer = SummaryWriter(log_dir=save_dir)

    # 读取数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 修改数据增强策略（更适合CNN+ViT混合模型）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 扩大裁剪范围
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 增强颜色扰动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准归一化
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # 数据加载器
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    
    # 初始化模型
    if args.model_type == "fusion":
        # 加载预训练ViT
        base_vit = create_model(args.model_name, num_classes=args.num_classes, has_logits=False)
        # 构建融合模型
        model = MultiScaleFusionViT(
            num_classes=args.num_classes,
            vit_model=base_vit
        ).to(device)
    else:
        base_vit = create_model(args.model_name, num_classes=1000, has_logits=True)
        model = base_vit.to(device)
        
    print('vit_model is %s'%(args.model_name))

    # 修改权重加载方式
    if args.weights != "":
        weights_dict = torch.load(args.weights, map_location=device)
        
        # 适配多级融合模型的权重加载
        if args.model_type == "fusion":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
            print("Loaded ViT weights successfully")
        else:
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            del_keys = ['head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
            


    
    
            
    # 修改冻结策略
    if args.freeze_layers:
        # 冻结ViT所有参数
        for name, param in model.named_parameters():
            if "vit." or "cnn_stages." in name:
                param.requires_grad_(False)
        for name, param in model.named_parameters():
            if "adapter."  in name:
                param.requires_grad_(True)
        for name, param in model.vit.named_parameters():
            if "head" in name or "pre_logits" in name:
                param.requires_grad_(True)
    '''
        for name, module in model.vit.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad_(True)
    '''
                    
                    
   
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad_
            print(f"Training: {name}")

        
    # 修改优化器设置（只训练非ViT部分）
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        pg,
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    # ... 后续代码保持不变 ...
        # 带预热的余弦退火调度器
    warmup_epochs = 5  # 5个epoch预热
    
        # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    freezing_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"freezing parameters: {freezing_params/1e6:.2f}M")

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热阶段
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火阶段（保留你原有的lrf参数）
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - args.lrf) + args.lrf

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
        # 初始化记录列表
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # 训练和验证循环
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        # 记录每个 epoch 的结果
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # 写入 TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存模型权重
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 训练完成后绘制曲线并保存
    plot_loss_acc(train_loss_list, val_loss_list, train_acc_list, val_acc_list, save_dir)

    print(f"Training completed! Results saved to {save_dir}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,default="./dataset1")
    parser.add_argument('--model-name', default='vit_base_patch16_224', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_1K.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers',action='store_true')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    
    # 在parser中添加模型类型参数
    parser.add_argument('--model-type', type=str, default='fusion',
                        choices=['original', 'fusion'],
                        help='model type: original ViT or fusion model')

    opt = parser.parse_args()

    main(opt)
