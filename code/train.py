import logging
import argparse
from statistics import mode

from pipeline import SparseConvSegmentation
from model import SparseEncDec
from dataset import MySemantic3D
import open3d._ml3d as ml3d
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    parser.add_argument('--data_path',
                        help='path to data.npy',
                        required=True)
    parser.add_argument('--ckpt_path',
                        help='path to saved checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args

def train(args,cfg):
    # Initialize the training by passing parameters
    dataset = MySemantic3D(args.data_path, use_cache=True, **cfg.dataset)
    model = SparseEncDec(dim_input=3,**cfg.model)
    pipeline = SparseConvSegmentation(model=model, dataset=dataset,**cfg.pipeline)
    pipeline.run_train()


if __name__ == '__main__':
    args = parse_args()
    cfg_file = "./config.yml"
    cfg = ml3d.utils.Config.load_from_file(cfg_file)
    train(args,cfg)