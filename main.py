"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
# import torch.multiprocessing as mp
# if __name__ == '__main__':
#     mp.set_start_method('spawn')

import argparse

from utils.config import *

import torch
import random
import numpy as np


from agents import *

#os.system("taskset -p -c 0 %d" % (os.getpid()))

#os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
# os.system("taskset -p -c 0-7,16-23 %d" % (os.getpid()))
# os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")

    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')

    arg_parser.add_argument('--mode', type=str, default='test')
    arg_parser.add_argument('--log_time_trained', type=str, default='0')

    arg_parser.add_argument('--num_agents', type=int, default=10)
    arg_parser.add_argument('--map_w', type=int, default=20)
    arg_parser.add_argument('--map_density', type=int, default=1)
    arg_parser.add_argument('--map_type', type=str, default='map')

    arg_parser.add_argument('--trained_num_agents', type=int, default=10)
    arg_parser.add_argument('--trained_map_w', type=int, default=20)
    arg_parser.add_argument('--trained_map_density', type=int, default=1)
    arg_parser.add_argument('--trained_map_type', type=str, default='map')

    arg_parser.add_argument('--nGraphFilterTaps', type=int, default=0)
    arg_parser.add_argument('--hiddenFeatures', type=int, default=0)

    arg_parser.add_argument('--num_testset', type=int, default=100)
    arg_parser.add_argument('--test_epoch', type=int, default=0)
    arg_parser.add_argument('--lastest_epoch', action='store_true', default=False)
    arg_parser.add_argument('--best_epoch', action='store_true', default=False)
    arg_parser.add_argument('--con_train', action='store_true', default=False)
    arg_parser.add_argument('--test_general', action='store_true', default=False)
    arg_parser.add_argument('--train_TL', action='store_true', default=False)
    arg_parser.add_argument('--Use_infoMode', type=int, default=0)
    arg_parser.add_argument('--log_anime', action='store_true', default=False)
    arg_parser.add_argument('--rate_maxstep', type=int, default=2)
    arg_parser.add_argument('--commR', type=int, default=6)
    arg_parser.add_argument('--map_noise_prob', type=float, default=None)
    arg_parser.add_argument('--map_shift_units', type=int, default=None)
    arg_parser.add_argument('--feature_noise_std', type=float, default=None)
    arg_parser.add_argument('--move_noise_std', type=float, default=None)
    arg_parser.add_argument('--comm_dropout_param', type=float, default=None)
    arg_parser.add_argument('--sybil_attack_count', type=int, default=None)
    arg_parser.add_argument('--rogue_agent_count', type=int, default=None)
    np.random.seed(1337)
    random.seed(1337)

    args = arg_parser.parse_args()
    # parse the config json file
    config = process_config(args)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)

    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
