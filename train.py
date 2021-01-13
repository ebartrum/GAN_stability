import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn, autograd
import hydra
from gan_training import utils
from gan_training.train import Trainer
from gan_training.logger import Logger
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import get_parameter_groups
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F
from collections import OrderedDict
from gan_training.metrics import inception_score
from pytorch_lightning.callbacks import ModelCheckpoint

class LM(pl.LightningModule):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        # Dataset
        self.train_dataset = get_dataset(
            name=cfg.data.type,
            data_dir=cfg.data.train_dir,
            size=cfg.data.img_size,
            lsun_categories=cfg.data.lsun_categories_train
        )

        # Number of labels
        self.nlabels = min(len(self.train_dataset.classes), cfg.data.nlabels)
        self.sample_nlabels = min(self.nlabels, cfg.train.sample_nlabels)

        # Create models
        self.generator = instantiate(cfg.generator)
        self.discriminator = instantiate(cfg.discriminator)
        self.gan_type = cfg.train.gan_type
        self.reg_type = cfg.train.reg_type
        self.reg_param = cfg.train.reg_param

        # Distributions
        self.ydist = get_ydist(self.nlabels)
        self.zdist = get_zdist(cfg.z_dist.type, cfg.z_dist.dim)

        # Save for tests
        ntest = cfg.train.batch_size
        x_real, self.ytest = utils.get_nsamples(self.train_dataloader(), ntest)
        self.ytest.clamp_(None, self.nlabels-1)
        self.ztest = self.zdist.sample((ntest,))

        # Evaluator
        self.evaluator = Evaluator(self.generator, self.zdist, self.ydist,
                              batch_size=cfg.train.batch_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.nworkers,
                shuffle=True, pin_memory=True, sampler=None, drop_last=True
        )
        
    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optimizer = self.cfg.train.optimizer
        lr_g = self.cfg.train.lr_g
        lr_d = self.cfg.train.lr_d
        equalize_lr = self.cfg.train.equalize_lr

        if equalize_lr:
            g_gradient_scales = getattr(self.generator, 'gradient_scales', dict())
            d_gradient_scales = getattr(self.discriminator, 'gradient_scales', dict())

            g_params = get_parameter_groups(self.generator.parameters(),
                                            g_gradient_scales,
                                            base_lr=lr_g)
            d_params = get_parameter_groups(self.discriminator.parameters(),
                                            d_gradient_scales,
                                            base_lr=lr_d)
        else:
            g_params = self.generator.parameters()
            d_params = self.discriminator.parameters()

        # Optimizers
        if optimizer == 'rmsprop':
            g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
            d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
        elif optimizer == 'adam':
            g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0., 0.99), eps=1e-8)
            d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0., 0.99), eps=1e-8)
        elif optimizer == 'sgd':
            g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
            d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

        # Learning rate anneling
        g_scheduler = optim.lr_scheduler.StepLR(
                g_optimizer,
                step_size=self.cfg.train.lr_anneal_every,
                gamma=self.cfg.train.lr_anneal)
        d_scheduler = optim.lr_scheduler.StepLR(
                d_optimizer,
                step_size=self.cfg.train.lr_anneal_every,
                gamma=self.cfg.train.lr_anneal)

        return (
           {'optimizer': g_optimizer, 'lr_scheduler': g_scheduler, 'frequency': 1},
           {'optimizer': d_optimizer, 'lr_scheduler': d_scheduler, 'frequency': self.cfg.train.n_critic} # TODO: correct this
       )

    def on_epoch_end(self):
        print('Creating images to log...')
        ztest, ytest = self.ztest.to(self.device), self.ytest.to(self.device)
        x = self.create_samples(ztest, ytest)
        self.logger.experiment.add_images('Generated samples', x, self.current_epoch)

    def training_step(self, batch, batch_nb, optimizer_idx):
        x_real, y = batch
        y.clamp_(None, self.nlabels-1)
        z = self.zdist.sample((self.cfg.train.batch_size,)).to(self.device)
        if optimizer_idx == 0:
            d_loss = self.discriminator_step(x_real, y, z)
            self.log('d_loss', d_loss)
            return d_loss
        elif optimizer_idx == 1:
            g_loss = self.generator_step(y, z)
            self.log('g_loss', g_loss)
            return g_loss

    def discriminator_step(self, x_real, y, z):
        with torch.no_grad():
            x_fake = self.generator(z, y)

        x_real.requires_grad_()
        d_real = self.discriminator(x_real, y)
        d_fake = self.discriminator(x_fake, y)
        d_loss_real = self.compute_loss(d_real, 1)
        d_loss_fake = self.compute_loss(d_fake, 0)

        if self.reg_type == 'real':
            reg = self.compute_grad2(d_real, x_real).mean()
        else:
            raise NotImplementedError(f"reg type {self.reg_type} not implemented")

        d_loss = (d_loss_real + d_loss_fake + self.reg_param*reg)
        return d_loss

    def generator_step(self, y, z):
        assert(y.size(0) == z.size(0))
        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        g_loss = self.compute_loss(d_fake, 1)
        return g_loss

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError(f"gan type {self.gan_type} not implemented")
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def compute_inception_score(self, inception_nsamples):
        imgs = []
        while(len(imgs) < inception_nsamples):
            ztest = self.zdist.sample((self.cfg.train.batch_size,)).to(self.device)
            ytest = self.ydist.sample((self.cfg.train.batch_size,)).to(self.device)

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )

        return score, score_std

    def create_samples(self, z, y=None):
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        x = self.generator(z, y)
        return x

@hydra.main(config_name="config")
def train(cfg: DictConfig) -> None:
    lm = LM(cfg)
    tb_logger = pl.loggers.TensorBoardLogger(
            save_dir='logs', name="my_model",
            default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor='d_loss', filename='model-{epoch:02d}-{loss:.2f}')
    trainer = pl.Trainer(gpus=1, logger=tb_logger,
            max_epochs=1000, callbacks=[checkpoint_callback])     
    trainer.fit(lm)

if __name__ == "__main__":
    train()
