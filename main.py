import json
import os

import shutil
import torch
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

from data.data_manager import DataManager
from trainer import Trainer
from utils.data_logs import save_logs_about
import utils.losses as loss_functions
from torch.utils.tensorboard import SummaryWriter
from networks.net import Net


def main():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config['device'])

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    logs_writer = SummaryWriter(os.path.join('runs', config['exp_name']))

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(config['device'])

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 3)
    model = model.to(config['device'])
    model = torch.nn.Sequential(*[model, torch.nn.Sigmoid()])

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))
    # shutil.copy(model.get_path(), os.path.join(config['exp_path'], config['exp_name']))

    criterion = torch.nn.BCELoss()
    #criterion = getattr(loss_functions, config['loss_function'])

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'])

    data_manager = DataManager(config)
    train_loader,eval_dataloader ,test_loader = data_manager.get_train_eval_test_dataloaders()

    trainer = Trainer(model, train_loader,eval_dataloader, criterion, optimizer, lr_scheduler, logs_writer, config)

    trainer.train()

    trainer.test_net(test_loader)


if __name__ == '__main__':
    main()
