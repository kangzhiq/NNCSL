# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

# import torch.multiprocessing as mp

import pprint
import yaml

from src.nncsl_train import main as nncsl
from src.utils import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--sel', type=str,
    help='which script to run',
    choices=[
        'nncsl_train',
    ])


def process_main(rank, sel, fname, world_size, devices):
    import os

    import logging
    logging.basicConfig()
    logger = logging.getLogger()

    logger.info(f'called-params {sel} {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        if rank == 0:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

    if rank == 0:
        if not os.path.exists(params['logging']['folder']):
            os.makedirs(params['logging']['folder'])
        dump = os.path.join(params['logging']['folder'], f'params-{sel}.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))

    # -- make sure all processes correctly initialized torch-distributed
    logger.info(f'Running {sel} (rank: {rank}/{world_size})')

    # -- turn off info-logging for ranks > 0, otherwise too much std output
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    train_fn = {
        'nncsl_train': nncsl
    }[sel](params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    process_main(0, args.sel, args.fname, num_gpus, args.devices)

