import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn, autograd
import hydra
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import get_parameter_groups
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F

class LM(pl.LightningModule):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        checkpoint_dir = path.join(cfg.train.out_dir, 'chkpts')
        # Create missing directories
        if not path.exists(cfg.train.out_dir):
            os.makedirs(cfg.train.out_dir)
        if not path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Logger
        checkpoint_io = CheckpointIO(
            checkpoint_dir=checkpoint_dir
        )

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

        # Register modules to checkpoint
        # checkpoint_io.register_modules(
        #     generator=generator,
        #     discriminator=discriminator,
        #     g_optimizer=g_optimizer,
        #     d_optimizer=d_optimizer,
        # )

        # Get model file
        # model_file = cfg.train.model_file

        # Logger
        # logger = Logger(
        #     log_dir=path.join(cfg.train.out_dir, 'logs'),
        #     img_dir=path.join(cfg.train.out_dir, 'imgs'),
        #     monitoring=cfg.train.monitoring,
        #     monitoring_dir=path.join(cfg.train.out_dir, 'monitoring')
        # )

        # Distributions
        self.ydist = get_ydist(self.nlabels)
        self.zdist = get_zdist(cfg.z_dist.type, cfg.z_dist.dim)

        # Save for tests
        # ntest = cfg.train.batch_size
        # x_real, ytest = utils.get_nsamples(self.train_dataloader, ntest)
        # ytest.clamp_(None, nlabels-1)
        # ztest = zdist.sample((ntest,))
        # utils.save_images(x_real, path.join(cfg.train.out_dir, 'real.png'))

        # Test generator
        # if cfg.train.take_model_average:
        #     generator_test = copy.deepcopy(generator)
        #     checkpoint_io.register_modules(generator_test=generator_test)
        # else:
        #     generator_test = generator

        # Evaluator
        self.evaluator = Evaluator(self.generator, self.zdist, self.ydist,
                              batch_size=cfg.train.batch_size)

        # Train
        # tstart = t0 = time.time()

        # Load checkpoint if it exists
        # try:
        #     load_dict = checkpoint_io.load(model_file)
        # except FileNotFoundError:
        #     it = epoch_idx = -1
        # else:
        #     it = load_dict.get('it', -1)
        #     epoch_idx = load_dict.get('epoch_idx', -1)
        #     logger.load_stats('stats.p')

        # Reinitialize model average if needed
        # if (cfg.train.take_model_average
        #         and cfg.train.model_average_reinit):
        #     update_average(generator_test, generator, 0.)


        # Trainer
        # trainer = Trainer(
        #     self.generator, self.discriminator, g_optimizer, d_optimizer,
        #     gan_type=cfg.train.gan_type,
        #     reg_type=cfg.train.reg_type,
        #     reg_param=cfg.train.reg_param
        # )

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

        # toggle_grad(self.generator, True)
        # toggle_grad(self.discriminator, True)

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
           {'optimizer': d_optimizer, 'lr_scheduler': d_scheduler, 'frequency': self.cfg.train.n_critic}
       )

    def on_epoch_end(self):
        import ipdb;ipdb.set_trace()
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        # (i) Sample if necessary
        if (it % cfg.train.sample_every) == 0:
            print('Creating samples...')
            x = evaluator.create_samples(ztest, ytest)
            logger.add_imgs(x, 'all', it)
            for y_inst in range(sample_nlabels):
                x = evaluator.create_samples(ztest, y_inst)
                logger.add_imgs(x, '%04d' % y_inst, it)

        # (ii) Compute inception if necessary
        if cfg.train.inception_every > 0 and ((it + 1) % cfg.train.inception_every) == 0:
            inception_mean, inception_std = evaluator.compute_inception_score()
            logger.add('inception_score', 'mean', inception_mean, it=it)
            logger.add('inception_score', 'stddev', inception_std, it=it)

        # (iii) Backup if necessary
        if ((it + 1) % cfg.train.backup_every) == 0:
            print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % it, it=it)
            logger.save_stats('stats_%08d.p' % it)

        # (iv) Save checkpoint if necessary
        if time.time() - t0 > cfg.train.save_every:
            print('Saving checkpoint...')
            checkpoint_io.save(model_file, it=it)
            logger.save_stats('stats.p')
            t0 = time.time()

            if (restart_every > 0 and t0 - tstart > restart_every):
                exit(3)

    def training_step(self, batch, batch_nb, optimizer_idx):
        x_real, y = batch
        y.clamp_(None, self.nlabels-1)
        z = self.zdist.sample((self.cfg.train.batch_size,)).to(self.device)
        if optimizer_idx == 0:
            # Discriminator updates
            dloss = self.discriminator_step(x_real, y, z)
            # logger.add('losses', 'discriminator', dloss, it=it)
            # logger.add('losses', 'regularizer', reg, it=it)
        elif optimizer_idx == 1:
            # Generators updates
            gloss = self.generator_step(y, z)
            # logger.add('losses', 'generator', gloss, it=it)

            if cfg.train.take_model_average:
                update_average(generator_test, generator,
                               beta=cfg.train.model_average_beta)

        # Print stats
        # g_loss_last = logger.get_last('losses', 'generator')
        # d_loss_last = logger.get_last('losses', 'discriminator')
        # d_reg_last = logger.get_last('losses', 'regularizer')
        # print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
        #       % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))

    def discriminator_step(self, x_real, y, z):
        with torch.no_grad():
            x_fake = self.generator(z, y)

        d_real = self.discriminator(x_real, y)
        d_fake = self.discriminator(x_fake, y)
        dloss_real = self.compute_loss(d_real, 1)
        dloss_fake = self.compute_loss(d_fake, 0)
        # x_real.requires_grad_()
        # x_fake.requires_grad_()

        if self.reg_type == 'wgangp':
            reg = self.wgan_gp_reg(x_real, x_fake, y)
        elif self.reg_type == 'wgangp0':
            reg = self.wgan_gp_reg(x_real, x_fake, y, center=0.)
        elif self.reg_type == 'real':
            dloss_real.backward(retain_graph=True)
            reg = self.compute_grad2(d_real, x_real).mean()
        elif self.reg_type == 'fake':
            dloss_fake.backward(retain_graph=True)
            reg = self.compute_grad2(d_fake, x_fake).mean()
        elif self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            dloss_fake.backward(retain_graph=True)
            reg = self.compute_grad2(d_real, x_real).mean() +\
                    self.compute_grad2(d_fake, x_fake).mean()
        elif self.reg_type == 'none':
            reg = torch.tensor(0.)

        # Output
        dloss = (dloss_real + dloss_fake + self.reg_param*reg)

        return dloss

    def generator_step(self, y, z):
        assert(y.size(0) == z.size(0))
        x_fake = self.generator(z, y)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)
        return gloss

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError
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

@hydra.main(config_name="config")
def train(cfg: DictConfig) -> None:
    lm = LM(cfg)
    trainer = pl.Trainer(gpus=1, logger=None,
            max_epochs=10)  
    trainer.fit(lm)

if __name__ == "__main__":
    train()
