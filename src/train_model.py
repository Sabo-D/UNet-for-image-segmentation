from itertools import accumulate

import torch
from torch import nn
import copy
import time
from tqdm import tqdm
import sys
from src.utils import compute_iou
from datetime import datetime
import pandas as pd

def model_train(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.load_state_dict(torch.load('D:\AA_Pycharm_Projs\\UNet\outputs\checkpoints\\best_model_2025-03-20_23-22.pth'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    '''
    CrossEntropyLoss(logits, targets)
    logits: raw logits (B,C,H,W) 在其内部会自动softmax->（B,H,W）
    targets: 类别索引 (B,H,W) 且必须是longTensor
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_all, val_loss_all = [], []  # 记录每个epoch的loss（平均）
    train_mean_iou_all, val_mean_iou_all = [], []
    # train_pre_class_iou_all, val_pre_class_iou_all = [], []
    best_loss = 1000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        train_loss, val_loss = 0.0, 0.0  # epoch的loss总和
        train_num, val_num = 0, 0  # epoch的num个数
        train_mean_iou, val_mean_iou = 0.0, 0.0
        # train_pre_class_iou, val_pre_class_iou = 0.0, 0.0

        since = time.time()

        model.train()
        for images, segments in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", file=sys.stdout):
            images, segments = images.to(device), segments.to(device)

            b_num = images.size(0)  # batch里的样本数量

            optimizer.zero_grad()
            outputs = model(images)  # (B, C, H, W) C=21
            loss = criterion(outputs, segments)  # 每个batch的平均loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_num  # 每个batch的loss相加（epoch的loss总和）
            mean_iou, iou_pre_class = compute_iou(outputs, segments, model.out_channels)
            train_mean_iou += mean_iou * b_num
            # train_pre_class_iou += iou_pre_class * b_num
            train_num += b_num  # 训练总的样本数量

        with torch.no_grad():
            model.eval()
            for images, segments in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", file=sys.stdout):
                images, segments = images.to(device), segments.to(device)
                segments = segments.squeeze(1)
                segments = segments.long()
                b_num = images.size(0)

                outputs = model(images)
                loss = criterion(outputs, segments)

                val_loss += loss.item() * b_num
                mean_iou, iou_pre_class = compute_iou(outputs, segments, model.out_channels)
                val_mean_iou += mean_iou * b_num
                # val_pre_class_iou += iou_pre_class * b_num
                val_num += b_num

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_mean_iou_all.append(train_mean_iou / train_num)
        val_mean_iou_all.append(val_mean_iou / val_num)
        # train_pre_class_iou_all.append(train_pre_class_iou / train_num)
        # val_pre_class_iou_all.append(val_pre_class_iou / val_num)

        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{} Train loss:{:.4f}  Train mean iou:{:.4f}  '.format(epoch + 1, train_loss_all[-1], train_mean_iou_all[-1]))
        print('{} Val loss:{:.4f}    Val mean iou:{:.4f}'.format(epoch + 1, val_loss_all[-1], val_mean_iou_all[-1]))
        print('Training time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_path = f'D:\AA_Pycharm_Projs\\UNet\outputs\checkpoints\\best_model_{current_time}.pth'
    torch.save(best_model_wts, model_path)

    train_process = pd.DataFrame(data={
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_loss_all,
        "train_mean_iou": train_mean_iou_all,
        "val_loss": val_loss_all,
        "val_mean_iou": val_mean_iou_all,
    })
    train_process = train_process.round(4)
    log_path = f'D:\AA_Pycharm_Projs\\UNet\outputs\logs\\train_logs\\logs_{current_time}.csv'
    train_process.to_csv(log_path)
    print(f"best loss{best_loss:.4f}")
    print('训练过程已成功保存')

    return train_process



