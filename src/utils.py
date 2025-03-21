import torch
from matplotlib import pyplot as plt
from datetime import datetime

from PIL import Image

def plot_iou_loss(train_process, type):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if type == 'train':
        plt.plot(train_process.epoch, train_process.train_loss,'ro-', label="train_loss")
        plt.plot(train_process.epoch, train_process.val_loss,'bx-', label="val_loss")
    if type == 'test':
        plt.plot(train_process.epoch, train_process.test_loss,'ro-', label="test_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('iou')
    if type == 'train':
        plt.plot(train_process.epoch, train_process.train_mean_iou, 'ro-', label="train_iou")
        plt.plot(train_process.epoch, train_process.val_mean_iou, 'bx-', label="val_iou")
    if type == 'test':
        plt.plot(train_process.epoch, train_process.test_mean_iou, 'ro-', label="test_iou")
    plt.legend()

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_path = None

    if type == 'train':
        out_path = f'D:\AA_Pycharm_Projs\\UNet\outputs\logs\\train_logs\\loss_iou_{current_time}'
    if type == 'test':
        out_path = f'D:\AA_Pycharm_Projs\\UNet\outputs\logs\\test_logs\\loss_iou_{current_time}'
    plt.savefig(out_path + '.png')
    plt.show()


def compute_iou(preds, targets, num_classes, smooth=1e-6):
    """
    计算多分类任务的 IoU（每个类别单独计算 IoU，然后求平均）

    参数:
    preds (tensor): 模型输出 logits，形状为 (N, C, H, W)
    targets (tensor): 真实标签，形状为 (N, 1, H, W)
    num_classes (int): 类别数
    smooth (float): 平滑系数避免除零，默认 1e-6

    返回:
    mean_iou (float): 所有类别 IoU 的平均值
    iou_per_class (list): 每个类别的 IoU
    """
    # 获取每个像素的预测类别 (argmax 选择通道最大值的索引作为类别)
    pred_classes = torch.argmax(preds, dim=1)  # (N, H, W)

    targets = targets.squeeze(1)  # 去掉目标标签的通道维度 (N, H, W)

    iou_per_class = []

    for class_idx in range(num_classes):
        # 创建当前类别的二值 mask
        pred_mask = (pred_classes == class_idx).float()  # 预测的当前类别区域
        target_mask = (targets == class_idx).float()  # 真实的当前类别区域

        # 计算交集和并集
        intersection = (pred_mask * target_mask).sum(dim=(1, 2))  # 按样本计算 (N,)
        union = pred_mask.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2)) - intersection

        # 计算 IoU（防止除零）
        iou = (intersection + smooth) / (union + smooth)

        # 计算每个类别的 IoU 均值
        iou_per_class.append(iou.mean().item())

    # 计算所有类别的 IoU 平均值
    mean_iou = sum(iou_per_class) / num_classes

    return mean_iou, iou_per_class  # 返回均值 IoU 和每个类别的 IoU


def test_iou():
    # 假设 preds 是模型输出的 raw logits，形状为 (N, C, H, W)
    N, C, H, W = 2, 3, 4, 4  # 2 个样本，3 个类别，4x4 图像
    preds = torch.tensor([[[[2.0, 1.5], [0.5, 3.0]],  # 类别 0
                           [[1.0, 2.5], [2.5, 1.0]],  # 类别 1
                           [[0.5, 1.0], [2.0, 0.5]]]], dtype=torch.float32)  # 类别 2

    targets = torch.tensor([[[0, 1], [1, 0]],  # 真实标签类别
                            [[1, 0], [0, 1]]], dtype=torch.int64)  # 0: 类别 0, 1: 类别 1
    mean_iou, iou_per_class = compute_iou(preds, targets, num_classes=C)
    # 输出结果
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU per class: {iou_per_class}")

if __name__ == '__main__':
    test_iou()
