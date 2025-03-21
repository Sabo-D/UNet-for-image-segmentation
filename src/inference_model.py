
from tqdm import tqdm
import sys
import torch
import time
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
import numpy as np
from PIL import  Image

def outputs_to_prediction(outputs, origin_size):
    """
    outputs->单通道PIL
    :param outputs: (B,C,H,W)
    :param height:
    :param width:
    :return:
    """
    outputs = outputs.squeeze(0)  # (C, H, W) tensor
    prediction = torch.argmax(outputs, dim=0)  # (H, W) tensor

    prediction = prediction.cpu().numpy().astype(np.uint8)  #(H, W) ndarray

    prediction = Image.fromarray(prediction)  # PIL

    prediction = prediction.resize(origin_size, Image.Resampling.NEAREST)

    return prediction

def prediction_to_color_image(prediction):
    """
    单通道PIL->三通道PIL
    :param prediction:
    :return:
    """
    VOC_PALETTE = np.array([
        [0, 0, 0],  # 背景 (0)
        [128, 0, 0],  # 类别1
        [0, 128, 0],  # 类别2
        [128, 128, 0],  # 类别3
        [0, 0, 128],  # 类别4
        [128, 0, 128],  # 类别5
        [0, 128, 128],  # 类别6
        [128, 128, 128],  # 类别7
        [64, 0, 0],  # 类别8
        [192, 0, 0],  # 类别9
        [64, 128, 0],  # 类别10
        [192, 128, 0],  # 类别11
        [64, 0, 128],  # 类别12
        [192, 0, 128],  # 类别13
        [64, 128, 128],  # 类别14
        [192, 128, 128],  # 类别15
        [0, 64, 0],  # 类别16
        [128, 64, 0],  # 类别17
        [0, 192, 0],  # 类别18
        [128, 192, 0],  # 类别19
        [0, 64, 128]  # 类别20
    ], dtype=np.uint8)

    prediction_ndarray = np.array(prediction)  # (H, W)
    color_image = VOC_PALETTE[prediction_ndarray]  # (H, W, 3)
    image = Image.fromarray(color_image)

    return image

def model_inference(model, inference_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    since = time.time()
    with torch.no_grad():
        model.eval()
        for image, image_name, origin_size in tqdm(inference_dataloader, desc=f"Inferencing", file=sys.stdout):
            image = image.to(device)  # (B,C,H,W)
            image_name = image_name[0]

            outputs = model(image)  # (B, C, H, W) C=21 B =1
            image = outputs_to_prediction(outputs, origin_size)
            image = prediction_to_color_image(image)
            image.save(f'D:\AA_Pycharm_Projs\\UNet\outputs\inference\\{image_name}.png')


    time_elapsed = time.time() - since
    print('Inferencing time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('推理过程已成功保存')

    return 0

