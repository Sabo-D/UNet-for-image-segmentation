import torch
import src.train_model as train_model
import src.utils as utils
import src.UNet as model
import src.dataset as dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, test_dataloader = dataset.train_val_test_dataloader()
    model = model.UNet(in_channels=3, out_channels=21).to(device)
    train_process = train_model.model_train(model, train_dataloader, val_dataloader, num_epochs=50)
    plot = utils.plot_iou_loss(train_process, 'train')


