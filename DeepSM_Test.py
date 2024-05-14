import numpy as np
import os
import torch
import time
from functions import DeepSM_test

from torchsummary import summary
from matplotlib import pyplot as plt
from torch.autograd import Variable
from scipy.io import loadmat
import torch.nn as nn
from torchstat import stat


from torch.utils.data import Dataset, DataLoader
from DeepSM_Model import Deep_SM

from DataGeneration import test_data_preparaton
from NetworkConfig import DeepSM_config




model_saving_path = '/home/admin1/JinganWang/Denoising/PytorchCode/ResUnetBiLSTM/ResUnetBiLSTMSPRMDeepSM/DeepSMTrainedModel.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
the_model = torch.load(model_saving_path)

model_untrained = Deep_SM()
params = torch.nn.utils.parameters_to_vector(model_untrained.parameters())
print(len(params))

the_model = torch.nn.DataParallel(the_model, device_ids=[0, 1, 2, 3])


the_model.to(device)
the_model.eval()
torch.cuda.empty_cache()



class TestDataset(Dataset):
    def __init__(self):
        self.data = np.array(test_data_preparaton())
        self.x_data = self.data[0, :, :, :, :]
        self.y_data = self.data[1, :, :, :, :]
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def to_var(tensor):
    return Variable(tensor.cuda())
#


TestInputFolder = '/home/admin1/JinganWang/Denoising/PytorchCode/ResUnetBiLSTM/ResUnetBiLSTMSPRMDeepSM/TestDataAll/Input/'
TestLabelFolder = '/home/admin1/JinganWang/Denoising/PytorchCode/ResUnetBiLSTM/ResUnetBiLSTMSPRMDeepSM/TestDataAll/GT/'



image_size = 384
num_frames_input = 4
num = 1


test_input_set = np.zeros([image_size, image_size, num_frames_input])


testinput_list = loadmat(TestInputFolder + str(num) + '.mat')
for item in testinput_list.items():
    test_input_value = item[1]

test_input_set[:, :, :] = test_input_value[:, :, 0:num_frames_input]

test_input_set_reshape = test_input_set.reshape([1, 1, image_size, image_size, num_frames_input])
test_input_set_reshape = np.transpose(test_input_set_reshape, (0, 4, 1, 2, 3))


inputs = test_input_set_reshape
inputs = torch.tensor(inputs)



inputs_single_frame_norm_sum = np.zeros([image_size, image_size])
inputs_single_frame_norm_sum = torch.Tensor(inputs_single_frame_norm_sum)
for k in range(num_frames_input):
    inputs_single_frame = inputs[0, k, 0, :, :]
    inputs_single_frame_norm = (inputs_single_frame - inputs_single_frame.min()) / (inputs_single_frame.max() - inputs_single_frame.min())
    inputs_single_frame_norm_sum = inputs_single_frame_norm_sum + inputs_single_frame_norm


inputs_single_frame_norm_sum_norm = (inputs_single_frame_norm_sum - inputs_single_frame_norm_sum.min()) / (inputs_single_frame_norm_sum.max() - inputs_single_frame_norm_sum.min())
inputs_single_frame_norm_sum_norm = inputs_single_frame_norm_sum_norm.cpu().detach().numpy()

inputs_single_frame_norm = inputs_single_frame_norm.cpu().detach().numpy()
plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(inputs_single_frame_norm, cmap='gray')
plt.title('Single Frame')

plt.subplot(1, 4, 2)
plt.imshow(inputs_single_frame_norm_sum_norm, cmap='gray')
plt.title('Four Average')

# inputs = inputs.float().to(config.device)

# print(inputs.shape)
time_start = time.time()
Inputs = np.zeros([1, num_frames_input, 1, image_size, image_size])
Inputs = inputs

Inputs = torch.Tensor(Inputs)
Inputs = Inputs.float().to(DeepSM_config.device)

with torch.no_grad():
    [UnetResult, UnetLSTMResult] = the_model(Inputs)
time_end = time.time()
time_use = time_end - time_start
print(time_use)

test_result = UnetLSTMResult
results_single_frame_norm_sum = np.zeros([image_size, image_size])
results_single_frame_norm_sum = torch.Tensor(results_single_frame_norm_sum)
for k in range(num_frames_input):
    results_single_frame = test_result[0, k, 0, :, :]
    results_single_frame_norm = (results_single_frame - results_single_frame.min()) / (results_single_frame.max() - results_single_frame.min())
    results_single_frame_norm = results_single_frame_norm.cpu()
    results_single_frame_norm_sum = results_single_frame_norm_sum + results_single_frame_norm

results_single_frame_norm = results_single_frame_norm.cpu().detach().numpy()
results_single_frame_norm_sum = results_single_frame_norm_sum.cpu().detach().numpy()




plt.subplot(1, 4, 3)
plt.imshow(results_single_frame_norm_sum, cmap='gray')
plt.title('Deep-SM Result')


test_label_folder_list = os.listdir(TestLabelFolder)


num_test_label = 1
test_label_set = np.zeros([image_size, image_size, num_frames_input])


testlabel_list = loadmat(TestLabelFolder + str(num) + '.mat')
for item in testlabel_list.items():
    test_label_value = item[1]

test_label_set[:, :, :] = test_label_value[:, :, 0:num_frames_input]

test_label_set_reshape = test_label_set.reshape([1, 1, image_size, image_size, num_frames_input])
test_label_set_reshape = np.transpose(test_label_set_reshape, (0, 4, 1, 2, 3))


labels = test_label_set_reshape

labels = torch.tensor(labels)
labels = labels.float().to(DeepSM_config.device)


labels_single = labels[0, 3, 0, :, :]
labels_single_norm = (labels_single - labels_single.min()) / (labels_single.max() - labels_single.min())
labels_single_norm = labels_single_norm.cpu().detach().numpy()
plt.subplot(1, 4, 4)
plt.imshow(labels_single_norm, cmap='gray')
plt.title('GT')



plt.show()
a = 1

