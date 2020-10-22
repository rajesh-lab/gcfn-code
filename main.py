from __future__ import print_function

import math
import copy
import argparse
import sys
import datetime
import time
import pdb
import os


import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.distributions as dists

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils_models import VDE, OutcomeModel, permutation, evaluate
from utils_data import generate_data_additive_treatment_process, generate_data_multiplicative_treatment_process, XZETY_Dataset

dirpath = os.path.dirname(os.path.realpath(__file__)) # no end-slash in this name
torch.set_printoptions(precision=5, linewidth=140)

parser = argparse.ArgumentParser(description='The General Control Function Method')
parser.add_argument('--save_dir', type=str, default=os.path.realpath(__file__))
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--max_epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--max_out_epochs', type=int, default=20, metavar='N',
                    help='number of outcome epochs to train (default: 2)')
parser.add_argument('--no-fig', action='store_true',
                    default=False, help='saves figures')
parser.add_argument('--seed', type=int, default=1000, metavar='S',
                    help='random seed (default: 1000)')
parser.add_argument('--lambda_', type=float, default=0.3)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--prefix', type=str, default="DUMMY_RUN")
parser.add_argument('--pdb', type=bool, default=False)
parser.add_argument('--load_vde', type=str, default=None,
                    help='path to vde model you want to load')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--zhat_dim', type=int, default=50)
parser.add_argument('--t_dim', type=int, default=20)
parser.add_argument('--vde_d', type=int, default=0)
parser.add_argument('--vde_K', type=int, default=2)
parser.add_argument('--out_d', type=int, default=50)
parser.add_argument('--out_K', type=int, default=2)
parser.add_argument('--experiment', type=str, default='add')
parser.add_argument('--recon_likelihood', type=str, default='cat')
parser.add_argument('--gpus', type=int, default=1,
                    help='set None to disable gpu')


args = parser.parse_args()
args.cuda = args.gpus is not None and torch.cuda.is_available()
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.manual_seed(args.seed)

# example VDE loading
# args.load_vde = dirpath + '/ckpts/RUN_GCFN-epoch=18.ckpt'

# VDE learning rates
args.lr_enc = 1e-2
args.lr_dec = 1e-2
args.lr_qzhat = 1e-2
args.lr_lb = 1e-5

# OUT learning rates
args.lr_out = 1e-3
args.lr_beta = 1e-2
args.out_lr_lb = 1e-5

""" ========================= data variables and data loaders ========================= """
batch_size = args.batch_size

if args.experiment == 'add':
    x, z, eps, t, y = generate_data_additive_treatment_process(
        m=args.sample_size, alpha=args.alpha)
elif args.experiment == 'mult':
    x, z, eps, t, y = generate_data_multiplicative_treatment_process(
        m=args.sample_size, alpha=args.alpha)
else:
    assert False, "only add/mult experiments implemented in this version"


def get_shape(tensor):
    if tensor is None:
        return 0
    else:
        if len(tensor.shape) > 1:
            return tensor.shape[1]
        elif len(tensor.shape) == 1:
            return 1
        else:
            assert False, 'input has no dimensions'


# updating args with
args.d_x = get_shape(x)
args.d_e = get_shape(eps)
if args.vde_d == 0:
    args.vde_d = 100

full_dataset = XZETY_Dataset(z, eps, t, y, x)
m_train = int(0.8*args.sample_size)
m_val = int(0.1*args.sample_size)
m_test = args.sample_size - m_train - m_val

train_data, val_data, test_data = random_split(
    full_dataset, [m_train, m_val, m_test])
train_loader, val_loader, test_loader = DataLoader(train_data, batch_size=args.batch_size), DataLoader(
    val_data, batch_size=m_val), DataLoader(test_data, batch_size=m_test)

filename_prefix = "_{}_lambda_{}_zhatdim{}_bs{}_e{}_ss{}_seed{}_".format(args.prefix,
                                                                         str(args.lambda_).replace(
                                                                             '.', ''),
                                                                         args.zhat_dim,
                                                                         args.batch_size,
                                                                         args.max_epochs,
                                                                         args.sample_size,
                                                                         args.seed)
args.filename = filename_prefix

""" ========================= data variables and data loaders ========================= """


if args.pdb:
    pdb.set_trace()


def main():
    if args.load_vde is not None:
        vde_model = VDE.load_from_checkpoint(args.load_vde)
        print('-'*40)
        print('LOADED MODEL FROM {}'.format(args.load_vde))
        print('-'*40)
    else:
        # ------------
        # VDE training
        # ------------

        checkpoint_callback = ModelCheckpoint(
            filepath= dirpath + '/ckpts/',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix='VDE_MODEL'
        )
        vde_model = VDE(args)
        vde_trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs,
                                 logger=None, checkpoint_callback=checkpoint_callback)
        vde_trainer.fit(vde_model, train_loader, val_loader)

    # ------------
    # outcome model creation
    # ------------
    outcome_checkpoint_callback = ModelCheckpoint(
        filepath=dirpath + '/ckpts/',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='OUT_MODEL'
    )
    # change ce_fn here to change true causal effect
    out_model = OutcomeModel(args, vde_model=vde_model, ce_fn=None)
    # assert False, "figure out what to give here."
    train_x = train_data.dataset[train_data.indices][0]
    train_t = train_data.dataset[train_data.indices][3]
    train_eps = train_data.dataset[train_data.indices][2]

    vde_model_cuda = vde_model.cuda()
    zhat_samples = vde_model_cuda.sample_zhat_from_data((train_x.cuda(
    ) if train_x is not None else None, None, train_eps.cuda(), train_t.cuda(), None)).view(-1)
    zhat_samples = permutation(zhat_samples.detach())
    out_model.zhat = zhat_samples

    # ------------
    # outcome training
    # ------------
    outcome_trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_out_epochs,
                                 logger=None, checkpoint_callback=outcome_checkpoint_callback)
    outcome_trainer.fit(out_model, train_loader, val_loader)
    outcome_trainer.test(out_model, test_dataloaders=test_loader)

    effect_pred = out_model.predict_y_do_t(torch.randn(100)).detach().cpu()
    save_dict = {
        'effect_pred': effect_pred
    }

if __name__ == "__main__":
    main()
