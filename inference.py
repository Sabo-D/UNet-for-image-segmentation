import torch
import src.inference_model as inference
import src.UNet as model
import src.dataset as dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_dataloader = dataset.inference_dataloader()
    model = model.UNet(in_channels=3, out_channels=21).to(device)
    model.load_state_dict(torch.load('D:\AA_Pycharm_Projs\\UNet\outputs\checkpoints\\best_model_2025-03-21_23-42.pth'))
    inference.model_inference(model, inference_dataloader)




