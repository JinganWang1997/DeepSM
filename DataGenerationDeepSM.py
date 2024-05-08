from scipy.io import loadmat
import os
import numpy as np
from scipy import io
num_frames_input = 4
image_size = 192



######################## Train ################################################
def train_data_preparaton():

    TrainInputFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/TrainData/TrainInput/'
    TrainLabelFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/TrainData/TrainLabel/'

    train_input_folder_list = os.listdir(TrainInputFolder)
    train_input_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_train_input= len(train_input_folder_list)
    train_input_set = np.zeros([num_train_input, image_size, image_size, num_frames_input])

    for num in range(0, num_train_input):
        traininput_list = loadmat(TrainInputFolder + str(num + 1) + '.mat')
        # print(num)
        for item in traininput_list.items():
            train_input_value = item[1]
        train_input_set[num, :, :, :] = train_input_value

    train_input_set_reshape = train_input_set.reshape([1, num_train_input, image_size, image_size, num_frames_input])
    train_input_set_reshape = np.transpose(train_input_set_reshape, (1, 0, 2, 3, 4))
    train_input_set_reshape = np.transpose(train_input_set_reshape, (0, 4, 1, 2, 3))


    # Label
    train_label_folder_list = os.listdir(TrainLabelFolder)
    train_label_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_train_label= len(train_label_folder_list)
    train_label_set = np.zeros([num_train_label, image_size, image_size, num_frames_input])
    for num in range(0, num_train_label):
        trainlabel_list = loadmat(TrainLabelFolder + str(num + 1) + '.mat')
        # print(num)
        for item in trainlabel_list.items():
            train_label_value = item[1]
        train_label_set[num, :, :, :] = train_label_value

    train_label_set_reshape = train_label_set.reshape([1, num_train_label, image_size, image_size, num_frames_input])
    train_label_set_reshape = np.transpose(train_label_set_reshape, (1, 0, 2, 3, 4))
    train_label_set_reshape = np.transpose(train_label_set_reshape, (0, 4, 1, 2, 3))

    train_input_set = train_input_set_reshape
    train_label_set = train_label_set_reshape

    print('Train Input Shape', train_input_set.shape)
    print('Train Label Shape', train_label_set.shape)

    return train_input_set, train_label_set



######################### Valid ################################################
def valid_data_preparaton():

    ValidInputFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/ValidationData/ValidationInput/'
    ValidLabelFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/ValidationData/ValidationLabel/'

    valid_input_folder_list = os.listdir(ValidInputFolder)
    valid_input_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_valid_input= len(valid_input_folder_list)
    valid_input_set = np.zeros([num_valid_input, image_size, image_size, num_frames_input])

    for num in range(0, num_valid_input):
        validinput_list = loadmat(ValidInputFolder + str(num + 1) + '.mat')
        for item in validinput_list.items():
            valid_input_value = item[1]
        valid_input_set[num, :, :, :] = valid_input_value

    valid_input_set_reshape = valid_input_set.reshape([1, num_valid_input, image_size, image_size, num_frames_input])
    valid_input_set_reshape = np.transpose(valid_input_set_reshape, (1, 0, 2, 3, 4))
    valid_input_set_reshape = np.transpose(valid_input_set_reshape, (0, 4, 1, 2, 3))


     # Label
    valid_label_folder_list = os.listdir(ValidLabelFolder)
    valid_label_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_valid_label = len(valid_label_folder_list)
    valid_label_set = np.zeros([num_valid_label, image_size, image_size, num_frames_input])
    for num in range(0, num_valid_label):
        validlabel_list = loadmat(ValidLabelFolder + str(num + 1) + '.mat')
        # print(num)
        for item in validlabel_list.items():
            valid_label_value = item[1]
        valid_label_set[num, :, :, :] = valid_label_value

    valid_label_set_reshape = valid_label_set.reshape([1, num_valid_label, image_size, image_size, num_frames_input])
    valid_label_set_reshape = np.transpose(valid_label_set_reshape, (1, 0, 2, 3, 4))
    valid_label_set_reshape = np.transpose(valid_label_set_reshape, (0, 4, 1, 2, 3))

    valid_input_set = valid_input_set_reshape
    valid_label_set = valid_label_set_reshape

    print('Valid Input Shape', valid_input_set.shape)
    print('Valid Label Shape', valid_label_set.shape)

    return valid_input_set, valid_label_set


######################## Test ################################################
def test_data_preparaton():

    TestInputFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/TestData/TestInput/'
    TestLabelFolder = '/home/admin1/JinganWang/Denoising/50nmGNP/FourInput/Volume/Data_Sample/TestData/TestLabel/'

    test_input_folder_list = os.listdir(TestInputFolder)
    test_input_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_test_input = len(test_input_folder_list)
    test_input_set = np.zeros([num_test_input, image_size, image_size, num_frames_input])

    for num in range(0, num_test_input):
        testinput_list = loadmat(TestInputFolder + str(num + 1) + '.mat')
        for item in testinput_list.items():
            test_input_value = item[0]
        test_input_set[num, :, :, :] = test_input_value

    test_input_set_reshape = test_input_set.reshape([1, num_test_input, image_size, image_size, num_frames_input])
    test_input_set_reshape = np.transpose(test_input_set_reshape, (1, 0, 2, 3, 4))
    test_input_set_reshape = np.transpose(test_input_set_reshape, (0, 4, 1, 2, 3))



    # Label
    test_label_folder_list = os.listdir(TestLabelFolder)
    test_label_folder_list.sort(key=lambda x: int(x.split('.')[0]))
    num_test_label= len(test_label_folder_list)
    test_label_set = np.zeros([num_test_label, image_size, image_size, num_frames_input])
    for num in range(0, num_test_label):
        testlabel_list = loadmat(TestLabelFolder + str(num + 1) + '.mat')
        # print(num)
        for item in testlabel_list.items():
            test_label_value = item[1]
        test_label_set[num, :, :, :] = test_label_value

    test_label_set_reshape = test_label_set.reshape([1, num_test_label, image_size, image_size, num_frames_input])
    test_label_set_reshape = np.transpose(test_label_set_reshape, (1, 0, 2, 3, 4))
    test_label_set_reshape = np.transpose(test_label_set_reshape, (0, 4, 1, 2, 3))

    test_input_set = test_input_set_reshape
    test_label_set = test_label_set_reshape

    print('Test Input Shape', test_input_set.shape)
    print('Test Label Shape', test_label_set.shape)

    return test_input_set, test_label_set