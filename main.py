## Deep-SM Model for SPRM Image Sequence Denosing


import numpy as np
import matplotlib
import os

import torch

matplotlib.use('TkAgg')

from utils import build_logging
from torch.utils.data import Dataset
from NetworkConfig import DeepSM_config
from functions import DeepSM_train
from functions import DeepSM_valid
from functions import DeepSM_test
from DeepSM_Model import Deep_SM

from MSELoss import MSELoss

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob

matplotlib.use('Agg')

from DataGenerationDeepSM import train_data_preparaton
from DataGenerationDeepSM import valid_data_preparaton
from DataGenerationDeepSM import test_data_preparaton


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
ids = [0, 1, 2, 3]



class TrainDataset(Dataset):
    def __init__(self):
        self.data = np.array(train_data_preparaton())
        self.input_data = self.data[0, :, :, :, :]
        self.GT_data = self.data[1, :, :, :, :]
        self.len = self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.GT_data[index]

    def __len__(self):
        return self.len


class ValidDataset(Dataset):
    def __init__(self):
        self.data = np.array(valid_data_preparaton())
        self.input_data = self.data[0, :, :, :, :]
        self.GT_data = self.data[1, :, :, :, :]
        self.len = self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.GT_data[index]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        self.data = np.array(test_data_preparaton())
        self.input_data = self.data[0, :, :, :, :]
        self.GT_data = self.data[1, :, :, :, :]
        a = self.input_data
        self.len = self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index], self.GT_data[index]

    def __len__(self):
        return self.len

model_saving_path = '/home/admin1/JinganWang/Denoising/PytorchCode/ResUnetBiLSTM/ResUnetBiLSTMSPRMDeepSM/output/model/'


def main():
    print('~' * 30)
    print('Deep-SM Model Training')
    print('~' * 30)
    name = 'DeepSM_Batch_8_lr_1e-4_step_size=50_gamma=0.5'
    logger = build_logging(DeepSM_config)
    # model = ConvLSTM(config).to(config.device)

    model = Deep_SM(1, 1, channel_num=64, temporal=1000)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)


    criterion = torch.nn.MSELoss().to(DeepSM_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)


    train_dataset = TrainDataset()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=DeepSM_config.train_batch_size,
                              num_workers=DeepSM_config.num_workers,
                              shuffle=True,
                              pin_memory=True)

    valid_dataset = ValidDataset()
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=DeepSM_config.valid_batch_size,
                              num_workers=DeepSM_config.num_workers,
                              shuffle=True,
                              pin_memory=True)


    train_records = []
    valid_records = []
    test_records = []
    for epoch in range(DeepSM_config.epochs):
        epoch_records = DeepSM_train(DeepSM_config, logger, epoch, model, train_loader, criterion, optimizer, scheduler)
        # print('epoch_records', epoch_records)
        train_records.append(np.mean(epoch_records['loss']))
        epoch_records = DeepSM_valid(DeepSM_config, logger, epoch, model, valid_loader, criterion)
        valid_records.append(np.mean(epoch_records['loss']))
        # epoch_records = test(config, logger, epoch, model, test_loader, criterion)
        # test_records.append(np.mean(epoch_records['loss']))
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        # plt.plot(range(epoch + 1), test_records, label='test')
        plt.legend()
        plt.savefig(os.path.join(DeepSM_config.output_dir, '{}.png'.format(name)))
        plt.close()
        if epoch % 5 == 0:
            model_saving_path_update = model_saving_path + str(epoch)
            print(model_saving_path_update)
            torch.save(model, model_saving_path_update)




if __name__ == '__main__':
    main()
