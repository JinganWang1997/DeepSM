from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch
import numpy as np
import matplotlib

from torch.autograd import Variable
matplotlib.use('TKAgg')


cuda_available = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]


def to_var(tensor):
    return Variable(tensor.cuda())


torch.cuda.empty_cache()
def DeepSM_train(config, logger, epoch, model, train_loader, criterion, optimizer, scheduler):

    model.train()
    epoch_records = {'loss': []}
    num_batch = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.float().to(config.device)
        inputs = to_var(inputs)
        targets = targets.float().to(config.device)
        targets = to_var(targets)
        [ResUNet_Output, DeepSM_Output] = model(inputs)

        frame_num = inputs.shape
        frame_num = frame_num[1]
        loss_train = 0.0

        for t in range(frame_num):
            loss_train += (criterion(ResUNet_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

        for t in range(frame_num):
            loss_train += (criterion(DeepSM_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

        optimizer.zero_grad()
        loss_train.requires_grad_(True)
        loss_train.backward()
        optimizer.step()
        epoch_records['loss'].append(loss_train.item())

        if batch_idx and batch_idx % config.display == 0:
            logger.info('EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batch,
                                                                                epoch_records['loss'][-1], np.mean(epoch_records['loss'])))
    scheduler.step()
    return epoch_records


def DeepSM_valid(config, logger, epoch, model, valid_loader, criterion):
    model.eval()
    epoch_records = {'loss': []}
    num_batch = len(valid_loader)

    for batch_idx, (inputs, targets) in enumerate(valid_loader):

        inputs = inputs.float().to(config.device)
        inputs = to_var(inputs)
        targets = targets.float().to(config.device)
        targets = to_var(targets)
        with torch.no_grad():
            [ResUNet_Output, DeepSM_Output] = model(inputs)

        frame_num = inputs.shape
        frame_num = frame_num[1]
        loss_valid = 0.0
        for t in range(frame_num):
            loss_valid += (criterion(ResUNet_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

        for t in range(frame_num):
            loss_valid += (criterion(DeepSM_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

        epoch_records['loss'].append(loss_valid.item())
        if batch_idx and batch_idx % config.display == 0:
            logger.info('[V] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batch,
                                                                                      epoch_records['loss'][-1],
                                                                                      np.mean(epoch_records['loss'])))
    return epoch_records



def DeepSM_test(config, logger, epoch, model, test_loader, criterion):

    model.eval()
    epoch_records = {'loss': []}
    num_batch = len(test_loader)

    for batch_idx, (inputs, targets) in enumerate(test_loader):

        with torch.no_grad():

            inputs = inputs.float().to(config.device)
            inputs = to_var(inputs)
            targets = targets.float().to(config.device)
            targets = to_var(targets)
            [ResUNet_Output, DeepSM_Output] = model(inputs)

            frame_num = inputs.shape
            frame_num = frame_num[1]
            loss_test = 0.0

            for t in range(frame_num):
                loss_test += (criterion(ResUNet_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

            for t in range(frame_num):
                loss_test += (criterion(DeepSM_Output[:, t, ...], targets[:, t, ...]) / (frame_num * 2.0))

            epoch_records['loss'].append(loss_test.item())
            if batch_idx and batch_idx % config.display == 0:
                logger.info('[T] EP:{:03d}\tBI:{:05d}/{:05d}\tLoss:{:.6f}({:.6f})'.format(epoch, batch_idx, num_batch,
                                                                                    epoch_records['loss'][-1], np.mean(epoch_records['loss'])))

    return epoch_records