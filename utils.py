from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import logging
import os


def build_logging(DeepSM_model_config):
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M',
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        filename=os.path.join(DeepSM_model_config.log_dir, time.strftime("%Y%d%m_%H%M") + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)




