import json

import torchvision

from data.base_dataset import BaseDataset
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd

class DataManager:
    def __init__(self, config):
        self.config = config

    # def get_dataloader(self, path):
    #     dataset = BaseDataset(path, self.config)
    #
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=self.config['batch_size'],
    #         shuffle=True,
    #         pin_memory=True if self.config['device'] == 'cuda' else False
    #     )
    #     return dataloader
    #
    # def get_train_eval_dataloaders(self):
    #     np.random.seed(707)
    #
    #     dataset = BaseDataset()
    #     dataset_size = len(dataset)
    #
    #     ## SPLIT DATASET
    #     train_split = self.config['train_size']
    #     train_size = int(train_split * dataset_size)
    #     validation_size = dataset_size - train_size
    #
    #     ########### CURRENTLY DOING THIS, WHICH WORKS ###########
    #     indices = list(range(dataset_size))
    #     np.random.shuffle(indices)
    #     train_indices = indices[:train_size]
    #     temp = int(train_size + validation_size)
    #     val_indices = indices[train_size:temp]
    #
    #     ## DATA LOARDER ##
    #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    #     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    #
    #     train_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                                batch_size=self.config['batch_size'],
    #                                                sampler=train_sampler,
    #                                                pin_memory=True if self.config['device'] == 'cuda' else False)
    #
    #     validation_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                                     batch_size=self.config['batch_size'],
    #                                                     sampler=valid_sampler,
    #                                                     pin_memory=True if self.config['device'] == 'cuda' else False)
    #     return train_loader, validation_loader

    def get_train_eval_test_dataloaders(self):
        np.random.seed(707)

        train_transform = transforms.Compose([
    transforms.Resize((512, 416)),  # Resize the image while maintaining the aspect ratio
    transforms.ToTensor(),           # Convert the image to a tensor
    transforms.Normalize(            # Normalize the image
        mean=[0.485, 0.456, 0.406],   # Pre-defined mean values for normalization
        std=[0.229, 0.224, 0.225]     # Pre-defined standard deviation values for normalization
    )
])

        test_transform = transforms.Compose([
    transforms.Resize((512, 416)),  # Resize the image while maintaining the aspect ratio
    transforms.ToTensor(),           # Convert the image to a tensor
    transforms.Normalize(            # Normalize the image
        mean=[0.485, 0.456, 0.406],   # Pre-defined mean values for normalization
        std=[0.229, 0.224, 0.225]     # Pre-defined standard deviation values for normalization
    )
])



        main_dir = 'E:\ml-work\Ranzcr CLip\data\dataset'

        train_df = 'data/data_train.csv'
        test_df = 'data/data_test.csv'
        trainset = BaseDataset(main_dir,train_df , train_transform)
        testset = BaseDataset(main_dir, test_df, test_transform)


        dataset_size = len(trainset)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        #valid_split = self.config['valid_size']
        #test_split = self.config['test_size']

        train_size = int(train_split * dataset_size)
        valid_size = int(dataset_size - train_size)
        test_size = len(testset)

        ########### ESTABLISHING INDICES FOR DATALOADERS ###########
        indices = list(range(dataset_size))
        indices_test = list(range(test_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:(train_size + valid_size)] ####
        test_indices = indices_test[:]      #### schimbare

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=self.config['batch_size'],
                                                  sampler=test_sampler,
                                                  pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader, test_loader


