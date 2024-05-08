from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch


root_dir = os.path.join(os.getcwd(), '.')
print(root_dir)

class Config:
    gpus = [0, 1, 2, 3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_workers = 8 * len(gpus)
        # num_workers = 2
        # train_batch_size = 1
        # valid_batch_size = 2 * train_batch_size
        # test_batch_size = 2 * train_batch_size
        train_batch_size = 8
        valid_batch_size = 8
        test_batch_size = 8
    else:
        num_workers = 0
        train_batch_size = 1
        valid_batch_size = 1
        test_batch_size = 1
        # valid_batch_size = 2 * train_batch_size
        # test_batch_size = 2 * train_batch_size


    num_frames_input = 4
    num_frames_output = 4
    input_size = (192, 192)
    step_length = 0.01
    display = 10
    draw = 10

    epochs = 1000

    encoder = [('conv', 'leaky', 1, 16, 3, 1, 2),
             ('convlstm', '', 16, 16, 3, 1, 1),
             ('conv', 'leaky', 16, 32, 3, 1, 2),
             ('convlstm', '', 32, 32, 3, 1, 1),
             ('conv', 'leaky', 32, 64, 3, 1, 2),
             ('convlstm', '', 64, 64, 3, 1, 1),
             ('conv', 'leaky', 64, 128, 3, 1, 2),
             ('convlstm', '', 128, 128, 3, 1, 1),
             ('conv', 'leaky', 128, 256, 3, 1, 2),
             ('convlstm', '', 256, 256, 3, 1, 1)]
    decoder = [('deconv', 'leaky', 256, 128, 4, 1, 2),
               ('convlstm', '', 256, 128, 3, 1, 1),
               ('deconv', 'leaky', 128, 64, 4, 1, 2),
               ('convlstm', '', 128, 64, 3, 1, 1),
               ('deconv', 'leaky', 64, 32, 4, 1, 2),
               ('convlstm', '', 64, 32, 3, 1, 1),
               ('deconv', 'leaky', 32, 16, 4, 1, 2),
               ('convlstm', '', 32, 16, 3, 1, 1),
               ('deconv', 'leaky', 16, 16, 4, 1, 2),
               ('convlstm', '', 17, 16, 3, 1, 1),
               ('conv', 'sigmoid', 16, 1, 1, 0, 1)]




    data_dir = os.path.join(root_dir, 'Data')
    output_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join(output_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cache_dir = os.path.join(output_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

DeepSM_config = Config()