from __future__ import print_function
from argparse import Namespace
from utils_data import real_number_batch_to_one_hot_vector_bins

import sys
import datetime
import time
import pdb
import math
import copy
import argparse
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stat

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal
import torch.distributions as dists
import pytorch_lightning as pl




class KLayerNet(nn.Module):
    def __init__(self, D_in=2, D_out=2, D_H=5, K=2):
        super(KLayerNet, self).__init__()
        assert K > 0, 'KLayerNet must at least be 1 layer.'

        d = D_H
        self.d = d
        self.d_in = D_in
        self.d_out = D_out
        self.d_mid = D_H

        layers = [
            nn.Linear(D_in, d),  # the output is also a 'class'
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        ]
        for _ in range(K-1):
            layers.extend(
                (
                    nn.Linear(d, d),  # the output is also a 'class'
                    nn.BatchNorm1d(d),
                    nn.ReLU(inplace=True),
                )
            )

        layers.append(nn.Linear(d, D_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class qZhatModel(nn.Module):
    def __init__(self, zhat_dim, D_x=0):
        super(qZhatModel, self).__init__()
        assert D_x >= 0, "covariate dimension cannot be negative"

        self.d_x = D_x  # COVARIATE DIMENSION
        self.zhat_dim = zhat_dim  # CONTROL FUNCTION DIM; CATEGORICAL COUNT CURRENTLY

        if self.d_x > 0:
            # CHANGE THIS TO set q(zhat | x) model. currently a one layer net.
            self._q = KLayerNet(D_in=self.d_x, D_out=self.zhat_dim,
                                D_H=max(10, self.d_x//2), K=1)
        else:
            self.internal_cat_log_p = torch.rand(
                self.zhat_dim, requires_grad=True)

    def get_parameters(self):
        if self.d_x > 0:
            return self._q.parameters()
        return self.internal_cat_log_p

    def forward(self, cat_p, x=None):
        """
        cat_p  (m,d)
        return (m)
        """
        if self.d_x > 0:
            cat_log_q = self._q(x)
            assert cat_log_q.shape == (
                x.shape[0], self.zhat_dim), cat_log_q.shape
            return -(cat_p*F.log_softmax(cat_log_q, dim=1)).sum(dim=1).mean()
        else:
            return -(cat_p*F.log_softmax(self.internal_cat_log_p.view(1, -1), dim=1)).sum(dim=1).mean()


class VDE(pl.LightningModule):

    def __init__(self, hparams, dec_fn=None):

        super(VDE, self).__init__()
        # we mainly work with scalar treatments; extending to multivariate involves changing reconstruction likelihood.
        self.d_t = 1

        if type(hparams) == type(dict()):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.d = hparams.vde_d
        self.zhat_dim = hparams.zhat_dim
        self.t_dim = hparams.t_dim
        self.d_e = hparams.d_e
        self.d_x = hparams.d_x
        self.verbose = hparams.verbose

        # the following are only needed for categorical reconstruction likelihood
        self.r_min = None
        self.r_max = None

        self.recon_likelihood = hparams.recon_likelihood

        assert self.recon_likelihood in [
            'cat', 'norm'], 'Only reconstruction likelihoods [cat, norm] implemented'

        self.lambda_ = hparams.lambda_

        print('input dim', self.d_e + 1 + self.d_x + 1)

        if dec_fn is None:
            if hparams.experiment == 'add':
                self.dec_fn = lambda a, b: a + b
            elif hparams.experiment == 'mult':
                self.dec_fn = lambda a, b: a * b
            else:
                raise ValueError('ONLY add AND mult HANDLED IN CODE')
        else:
            self.dec_fn = dec_fn

        if self.recon_likelihood == 'cat':
            # NOTE CAT TAKES A LOT OF GPU MEMORY; TENSOR ARE EXPANDED TO ENSURE SPEED.
            self.dec_fn = 'general'
            #  THIS IS FOR A GENERAL DECODER
            self._dec_eps_zhat = KLayerNet(D_in= self.d_x + self.d_e + self.d_t + 1, D_out=1, D_H=self.d, K=self.hparams.vde_K)

            """
                NOTE: r_min and r_max define upper and lower limits of discretization of t.
                [-infty, r_min] and [r_max, infty] for 2 out of how many ever bins.
                The following setting works for a standard normal t.
                When using a different distribution please choose these appropriately.
                If chosen without care, one bin could be large and contain many treatment values all of which will be treated the same in VDE; this leads to GCFN estimating the same control function for all treatment values in this bin, for a fixed IV val.
                For example, if r_min is 0 for a [-1,0] uniform R.V. all the treatment values will be in the bin [-infty, r_min].
            """
            print(' USING r_min r_max for standard normal;')
            self.r_min = -3.5
            self.r_max = 3.5
        self._enc, self._dec_eps, self._dec_zhat, self._q_zhat_model = self.init_models()

        self.z_vec = 0.5 + torch.arange(0, self.zhat_dim, 1).float()
        self.t_vec = 0.5 + torch.arange(0, self.t_dim, 1).float()
        self.z_vec_sample = torch.eye(self.zhat_dim)

    def init_models(self):

        # the extra 1 dim corresponds to zhat
        _enc = KLayerNet(D_in=self.d_e + self.d_t + self.d_x + 1,
                         D_out=1, D_H=self.d, K=self.hparams.vde_K)

        _dec_eps = KLayerNet(D_in=self.d_e + self.d_x,
                             D_out=1, D_H=self.d, K=self.hparams.vde_K)

        _dec_zhat = KLayerNet(D_in=self.d_x + 1,
                              D_out=1, D_H=self.d, K=self.hparams.vde_K)

        _q_zhat_model = qZhatModel(zhat_dim=self.zhat_dim, D_x=self.d_x)

        return _enc, _dec_eps, _dec_zhat, _q_zhat_model

    def vde_loss(self, logit_cat_q_zhat, t_rho_pred, t):
        if self.recon_likelihood == 'cat':
            assert t_rho_pred.size() == (
                t.shape[0]*self.zhat_dim, self.t_dim), (t_rho_pred.shape, t.shape[0], self.zhat_dim, self.t_dim)

        if self.recon_likelihood == 'cat':
            t_target = real_number_batch_to_one_hot_vector_bins(
                t, self.t_dim).long()
        elif self.recon_likelihood == 'norm':
            t_target = t.clone()
        else:
            assert False, "model {} not implemented".format(
                self.recon_likelihood)

        t_target = t_target.view(-1, 1).repeat(1, self.zhat_dim).view(-1)

        # cat_q_zhat is indexed by i for each (t_i,eps_i,x_i)
        cat_q_zhat = F.softmax(logit_cat_q_zhat.view(-1, self.zhat_dim), dim=1)

        _recon_loss = self.recon_loss(t_target, t_rho_pred, cat_q_zhat)
        _mi_lb = self.KL_loss(cat_q_zhat)

        return _recon_loss + self.lambda_*_mi_lb, _recon_loss, _mi_lb

    def recon_loss(self, t_target, t_rho_pred, cat_q_zhat):

        if self.recon_likelihood == 'cat':
            t_per_zhat_loss = F.cross_entropy(
                t_rho_pred, t_target, reduction='none')
        elif self.recon_likelihood == 'norm':
            # gaussian reconstruction likelihood
            t_per_zhat_loss = F.mse_loss(
                t_rho_pred, t_target, reduction='none')
        else:
            assert False, "model {} not implemented".format(
                self.recon_likelihood)

        recon_loss_vec = cat_q_zhat*t_per_zhat_loss.view(-1, self.zhat_dim)

        return recon_loss_vec.mean()*self.zhat_dim

    def KL_loss(self, cat_q_zhat):
        cross_entropy_marginal_q_zhat = self._q_zhat_model(cat_q_zhat)
        conditional_enttropy_q_zhat = _E_neglog_prob_categorical(cat_q_zhat)
        mutual_info_lb = cross_entropy_marginal_q_zhat - conditional_enttropy_q_zhat

        return mutual_info_lb

    def semi_sup_loss(self, logit_cat_q_zhat, z, mask):
        cat_q_zhat = F.softmax(logit_cat_q_zhat.view(-1, self.zhat_dim), dim=1)
        z_obs = z[mask == 0]
        cat_q_zhat_obs = cat_q_zhat[mask == 0, :]
        return F.cross_entropy(cat_q_zhat_obs, z_obs, reduction='mean')

    def training_step(self, batch, batch_idx):
        _, _, _, t, _ = batch

        # encoder and decoder output
        logit_cat_q_zhat, t_rho_pred = self.forward(batch)

        loss, recon_loss, mi_lb = self.vde_loss(
            logit_cat_q_zhat, t_rho_pred, t)

        loss_dict = {'loss': loss, 'recon_loss': recon_loss, 'mi_lb': mi_lb}

        # print(' ---- TRAIN ---- batch idx {}  : loss {:.3f}, recon loss {:.3f}, MI lb {:.3f}'.format(batch_idx, loss.item(), recon_loss.item(), mi_lb.item()))

        self.log('train_loss', loss, prog_bar=True)
        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('mi_lb', mi_lb, prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx):
        _, _, _, t, _ = batch

        # encoder and decoder output
        logit_cat_q_zhat, t_rho_pred = self.forward(batch)

        loss, recon_loss, mi_lb = self.vde_loss(
            logit_cat_q_zhat, t_rho_pred, t)

        loss_dict = {'loss': loss, 'recon_loss': recon_loss, 'mi_lb': mi_lb}

        # print(self._q_zhat_model.get_parameters().detach())
        # print(' ---- VAL ---- batch idx {}  : loss {:.3f}, recon loss {:.3f}, MI lb {:.3f}'.format(batch_idx, loss.item(), recon_loss.item(), mi_lb.item()))

        self.log('val_loss', loss, on_step=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=True, prog_bar=True)
        self.log('mi_lb', mi_lb, on_step=True, prog_bar=True)

        return loss_dict

    def configure_optimizers(self):
        params = [
            {'params': self._enc.parameters(), 'lr': self.hparams.lr_enc},
            {'params': self._dec_eps.parameters(), 'lr': self.hparams.lr_dec},
            {'params': self._dec_zhat.parameters(), 'lr': self.hparams.lr_dec},
            {'params': self._q_zhat_model.get_parameters(), 'lr': self.hparams.lr_qzhat}
        ]
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer

    def encode(self, zhat, eps, t, x=None):
        """compute the logits of likelihood of q(zhat | t, eps)"""

        t_target = self.fix_t(real_number_batch_to_one_hot_vector_bins(
            t, self.t_dim, return_one_hot=False).float())

        if self.d_x > 0:
            if x is None:
                assert False, 'd_x>0 but no x in encode'

            enc_input = torch.cat([x.view(-1, self.d_x), eps.view(-1, self.d_e),
                                   t_target.view(-1, 1), zhat.view(-1, 1)], dim=1)
        else:
            enc_input = torch.cat(
                [eps.view(-1, self.d_e), t_target.view(-1, 1), zhat.view(-1, 1)], dim=1)

        # distribution over the categories of zhat
        logit_cat_q_zhat = self._enc(enc_input)

        return logit_cat_q_zhat

    def sample_from_cat_logits(self, cat_logits):
        cat_dist = dists.Categorical(logits=cat_logits)
        return cat_dist.sample().float()

    def sample_zhat_from_data(self, data):

        logit_cat_q_zhat = self.encode_from_data(data)
        cat_q_zhat = F.softmax(logit_cat_q_zhat.view(-1, self.zhat_dim), dim=1)

        zhat_samples = self.sample_from_cat_logits(cat_q_zhat).view(-1)
        assert zhat_samples.shape[0] == data[3].shape[0]

        return self.fix_z(zhat_samples)

    def decode(self, zhat, eps):
        """compute the logits of likelihood of p(t| zhat, eps)"""

        if self.dec_fn == 'general':
            # the following is general categorical t reconstruction.
            dec_input = torch.cat([zhat, eps.view(-1, 1)], dim=1)

            assert dec_input.shape[1] == 1 + self.d_e + self.d_x
            h_interim = self._dec_eps_zhat(torch.cat([dec_input.repeat(self.t_dim, 1),
                                             self.fix_t(self.t_vec.view(-1, 1).repeat(1, eps.shape[0]).view(-1, 1))],
                                            dim=1))

            return h_interim.view(self.t_dim, -1).t()

        else:
            # assuming a 1-d treatment
            phi_eps = self._dec_eps(eps.view(-1, self.d_e)).view(-1)
            # assuming a 1-d treatment
            phi_zhat = self._dec_zhat(zhat.view(-1, 1)).view(-1)

            assert phi_eps.shape == phi_zhat.shape, (
                phi_eps.shape, phi_zhat.shape)

            return self.dec_fn(phi_eps, phi_zhat)

    def fix_z(self, z):
        return (z - self.z_vec.mean())/10

    def fix_t(self, t):
        return (t - self.t_vec.mean())/10

    def expand_for_encoder(self, data):
        # data is always of the for covariates, confounder, IV, treatment, outcome
        x, _, eps, t, _ = data

        eps_expand = eps.view(-1, self.d_e).repeat(1, self.zhat_dim).view(-1)
        t_expand = t.view(-1, 1).repeat(1, self.zhat_dim).view(-1)
        if self.d_x > 0:
            # TODO: check this
            x_expand = x.repeat(1, self.zhat_dim).view(-1, self.d_x)
        else:
            x_expand = None

        zhat_expand = self.fix_z(self.z_vec.repeat(eps.shape[0]).view(-1, 1))

        return x_expand, zhat_expand, eps_expand, t_expand

    def encode_from_data(self, data):
        x_expand, zhat_expand, eps_expand, t_expand = self.expand_for_encoder(
            data)
        return self.encode(zhat_expand, eps_expand, t_expand, x_expand)

    def forward(self, data):
        # data is always of the for covariates, confounder, IV, treatment, outcome
        x, _, eps, t, _ = data

        # ENCODING STEP
        x_expand, zhat_expand, eps_expand, t_expand = self.expand_for_encoder(
            data)
        assert eps_expand.shape == (
            eps.shape[0]*self.zhat_dim,), (eps_expand.shape, eps.shape, self.zhat_dim, eps.shape[0]*self.zhat_dim)

        logit_cat_q_zhat = self.encode(zhat_expand, eps_expand, t_expand)

        assert zhat_expand.shape == (
            eps.shape[0]*self.zhat_dim, 1), (zhat_expand.shape, self.zhat_dim, eps.shape)

        """this parameter determines the reconstruction likelihood evaluation. For example, when reconstruction is categorical, t_pred_pre will be logits; when reconstruction is gaussian, t_pred_pre will be mu. This is determined in self.loss_function"""

        t_rho_pred = self.decode(zhat_expand, eps_expand)

        return logit_cat_q_zhat, t_rho_pred


def permutation(a):
    idx = torch.randperm(a.nelement())
    return a[idx]


class OutcomeModel(pl.LightningModule):

    def __init__(self, hparams, vde_model, ce_fn=None):
        super().__init__()

        if type(hparams) == type(dict()):
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.d = hparams.out_d
        self.out_K = hparams.out_K
        self.d_x = hparams.d_x
        self.ce_fn = ce_fn

        self.vde_model = vde_model

        # This assumes that zhat and treatment are both 1 dimensional
        self._out = KLayerNet(D_in=self.d_x + 1 + 1,
                              D_out=1, D_H=self.d, K=self.out_K)

        self.zhat = None

    def forward(self, t, zhat_samples, x=None):
        if self.d_x > 0:
            model_input = torch.cat(
                [x.view(-1, self.d_x), t.view(-1, 1), zhat_samples.view(-1, 1)], dim=1)
        else:
            model_input = torch.cat(
                [t.view(-1, 1), zhat_samples.view(-1, 1)], dim=1)
        y_pred = self._out(model_input)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, _, _, t, y = batch
        zhat_samples = self.vde_model.sample_zhat_from_data(batch)

        y_pred = self.forward(t, zhat_samples, x).view(-1)
        outcome_loss = F.mse_loss(y_pred, y)

        return outcome_loss

    def validation_step(self, batch, batch_idx):
        x, _, _, t, y = batch
        zhat_samples = self.vde_model.sample_zhat_from_data(batch)

        y_pred = self.forward(t, zhat_samples, x).view(-1)

        assert y_pred.shape == y.shape, (y_pred.shape, y.shape)
        outcome_loss = F.mse_loss(y_pred, y)

        _, effect_mse = evaluate(self.predict_y_do_t, t, x, self.ce_fn)

        loss_dict = {
            'loss': outcome_loss,
            'effect_mse': effect_mse
        }

        self.log('val_loss', outcome_loss, on_step=True, prog_bar=True)
        self.log('effect_mse', effect_mse, on_step=True, prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx):
        # THIS SAVES THE EFFECT PREDICTION ON THE TEST SET; SO ENSURE THE WHOLE TEST SET IS IN A SINGLE BATCH.
        x, _, _, t, y = batch
        zhat_samples = self.vde_model.sample_zhat_from_data(batch)

        y_pred = self.forward(t, zhat_samples, x).view(-1)

        assert y_pred.shape == y.shape, (y_pred.shape, y.shape)
        outcome_loss = F.mse_loss(y_pred, y)

        effect_pred, effect_mse = evaluate(self.predict_y_do_t, t, x, self.ce_fn)

        loss_dict = {
            'test_loss': outcome_loss,
            'test_effect_mse': effect_mse,
        }

        save_dict = {
            'test_loss': outcome_loss,
            'test_effect_mse': effect_mse,
            'test_effect_pred': effect_pred.detach().cpu()
        }
        torch.save(save_dict, self.hparams.save_dir + 'test_' + self.hparams.filename + '.dict')

        self.log('test_loss', outcome_loss, on_step=True, prog_bar=True)
        self.log('test_effect_mse', effect_mse, on_step=True, prog_bar=True)

        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self._out.parameters(), lr=self.hparams.lr_out)
        return optimizer

    def predict_y_do_t(self, t, x=None):

        if self.zhat is None:
            assert False, 'please create zhat using vde model on data of the form (x, _, eps, t, _)'

        y_do_tz_samples = []
        N_mc_samples_to_compute = 100
        zhat_sample = permutation(self.zhat).view(-1)[:N_mc_samples_to_compute]

        for idx in range(N_mc_samples_to_compute):
            _zhat = zhat_sample[idx].item()*t.new_ones(t.shape[0])
            y_do_tz_samples.append(self.forward(
                t, _zhat, x).view(-1, 1).detach())

        y_do_tz = torch.cat(y_do_tz_samples, dim=1)

        y_do_t_pred = y_do_tz.mean(dim=1).view(-1)

        assert y_do_t_pred.shape[0] == t.shape[0], (y_do_t_pred.shape, t.shape)

        return y_do_t_pred


def _E_neglog_prob_categorical(cat_p, cat_log_p=None):
    """ cat_p = (m,d), I compute E_z log p(z)
    return (m,) """
    cat_log_p = cat_log_p if cat_log_p is not None else torch.log(cat_p + 1e-6)
    return -(cat_log_p*cat_p).sum(dim=1).mean()


def _E_neglog_prob_categorical_from_samples(logit_cat_p, z_sample_indexes):
    cat_dist = dists.Categorical(logits=logit_cat_p)
    return -1*cat_dist.log_prob(z_sample_indexes).mean()


def one_hot(batch, depth):
    ones = torch.eye(depth)
    return ones.index_select(0, batch)


def unique_from_tensor(tensor1d, return_np=True):
    t, idx = np.unique(tensor1d.cpu().numpy(), return_counts=True)
    if return_np:
        return t, idx
    return torch.from_numpy(t), torch.from_numpy(idx)


def evaluate(outcome_model_predict_y_do_t, t, x=None, ce_fn=None):

    # restricted range of evaluation
    # -------------------
    # THIS CODE PART IS SPECIFIC TO OUR EXPERIMENTS
    t_eval = t.view(-1).clone()
    # dataset return empty tensor along dim 1
    x_eval = x[t_eval.abs() <= 1] if x is not None and x.shape[1] == 0 else None
    t_eval = t_eval[t_eval.abs() <= 1]
    # -------------------

    if ce_fn is None:
        # this is the effect in all our experiments
        effect = t_eval
    else:
        # ce_fn is expected to compute the true causal to compare against.
        effect = ce_fn(t_eval, x_eval)

    effect_pred = outcome_model_predict_y_do_t(
        t_eval, x_eval).view(-1).detach()

    # compute MSE
    effect_mse = (effect - effect_pred).pow(2).mean()

    # plt.figure()
    # plt.scatter(t_eval.cpu().numpy(), t_eval.cpu().numpy())
    # plt.scatter(t_eval.cpu().numpy(), effect_pred.cpu().numpy())
    # plt.savefig('/misc/vlgscratch4/RanganathGroup/aahlad/conf_iv/gcfn_code_submission_neurips2020/debug_dummy_wat.png')
    # plt.close()

    return effect_pred, effect_mse.item()
