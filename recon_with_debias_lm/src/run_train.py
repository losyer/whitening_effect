# coding: utf-8
import argparse
import json
import os
import sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from prepare_dataset import DataHandler
from trainer import Trainer
from global_utils import prepare_directory
import logging


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    log_format = '%(asctime)s [%(levelname)-8s] [%(module)s#%(funcName)s %(lineno)d] %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    handler.flush = sys.stdout.flush
    logger.addHandler(handler)

    return logger


logger = init_logger(None)


def get_sampler(split, args, shuffle=False, distributed=False, rank=0):
    if distributed:
        return DistributedSampler(split, num_replicas=args.n_gpu, rank=rank, shuffle=shuffle)
    else:
        return None


def initial_setup(args):
    # Setup result directory
    result_dest_name = args.result_dir
    result_dest = prepare_directory(result_dest_name+'/')

    with open(os.path.join(result_dest, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))
    print()
    print('###########')
    print('#Arguments#')
    print('###########')
    print(json.dumps(vars(args), sort_keys=True, indent=4), flush=True)
    logger.info("result dest: " + result_dest)

    return result_dest


def model_setup(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    lm_model = AutoModel.from_pretrained(args.model_name_or_path,
                                         output_hidden_states=True)
    lm_model_ref = AutoModel.from_pretrained(args.model_name_or_path,
                                             output_hidden_states=True)
    return tokenizer, lm_model, lm_model_ref


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initial setup
    result_dest = initial_setup(args)
    distributed = False
    rank = 0

    # Model setup
    tokenizer, lm_model, lm_model_ref = model_setup(args)

    # Data setup
    data_handler = DataHandler(args, tokenizer)
    data_handler.prepare_dataset()
    train_dataset = data_handler.train_data

    sampler = get_sampler(train_dataset,
                          args,
                          shuffle=False,
                          distributed=distributed,
                          rank=rank)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=False,
                                  batch_size=args.batch_size,
                                  sampler=sampler)

    selected_epochs = []
    if args.save_selected_epoch:
        selected_epochs = args.save_selected_epoch.split('_')
        selected_epochs = [int(epoch)-1 for epoch in selected_epochs]

    trainer = Trainer(lm_model,
                      lm_model_ref,
                      tokenizer,
                      train_dataloader,
                      device='cuda',
                      args=args,
                      epochs=args.epoch,
                      optimizer_class=torch.optim.Adam,
                      optimizer_params={'lr': args.lr},
                      weight_decay=0.0,
                      output_path=result_dest,
                      save_best_model=True,
                      max_grad_norm=999999.0,
                      use_amp=True,
                      rank=0,
                      save_per_epoch=False,
                      save_selected_epoch=selected_epochs,
                      )

    trainer.run()


def parse_args():
    parser = argparse.ArgumentParser()

    # Training parameter
    parser.add_argument('--epoch', dest='epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0)

    # Model parameter
    parser.add_argument('--model_name_or_path', type=str, default="")
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=768)
    parser.add_argument('--adv_freq_thresh', type=int, default=3000)
    parser.add_argument('--adv_lr', type=float, default=0.02)
    parser.add_argument('--adv_wdecay', type=float, default=1.2e-6)
    parser.add_argument('--adv_lambda', type=float, default=0.02)

    # Training flag
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--normalize', action='store_true')

    # Other flag
    parser.add_argument('--show_iter_progress_bar', action='store_true')

    # Data path
    # For training
    parser.add_argument('--corpus_path', type=str, default="")
    parser.add_argument('--result_dir', type=str, default='/tmp/')
    parser.add_argument('--freq_path', type=str, default="")
    parser.add_argument('--out_prefix', type=str, default="")

    # For saving
    parser.add_argument('--save_selected_epoch', type=str, default=None)

    parser.add_argument('--layer_index', type=str, default='12')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
