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
        train_batch_size = 32
        valid_batch_size = 32
        test_batch_size = 32
    else:
        num_workers = 0
        train_batch_size = 1
        valid_batch_size = 1
        test_batch_size = 1


    num_frames_input = 4
    num_frames_output = 4
    input_size = (192, 192)
    step_length = 0.01
    display = 10
    draw = 10

    epochs = 1000

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
