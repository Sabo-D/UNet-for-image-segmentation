from tqdm import tqdm
import sys
import torch
from torch import nn
import time
from datetime import datetime
import pandas as pd
import src.utils as utils

def model_test(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    test_loss, test_mean_iou = 0.0, 0.0
    test_num =0
    since = time.time()

    with torch.no_grad():
        model.eval()
        for images, segments in tqdm(test_dataloader, desc=f"Testing ", file=sys.stdout):
            images, segments = images.to(device), segments.to(device)
            segments = segments.squeeze(1)
            segments = segments.long()

            outputs = model(images)
            loss = criterion(outputs, segments)

            test_loss += loss.item()
            mean_iou, mean_iou_pre_class  = utils.compute_iou(outputs, segments, model.out_channels)
            test_mean_iou += mean_iou
            test_num += 1

    time_elapsed = time.time() - since
    test_loss = test_loss / test_num
    test_mean_iou = test_mean_iou / test_num

    print('Test loss:{:.4f}  Test mean iou:{:.4f}'.format( test_loss, test_mean_iou))
    print('Testing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    test_process = pd.DataFrame(data={
        'test_loss': [test_loss],
        'test_mean_iou': [test_mean_iou],
    })
    test_process = test_process.round(4)
    log_path = f'D:\AA_Pycharm_Projs\\UNet\outputs\logs\\test_logs\\logs_{current_time}.csv'
    test_process.to_csv(log_path)
    print('测试过程已成功保存')

    return test_process