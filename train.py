import argparse
import os
from os import path
import time
import copy
import torch
from torch import nn
import hydra
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    build_optimizers, build_lr_scheduler,
)
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
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

    device = torch.device("cuda:0" if is_cuda else "cpu")

    # Dataset
    train_dataset, nlabels = get_dataset(
        name=cfg['data']['type'],
        data_dir=cfg['data']['train_dir'],
        size=cfg['data']['img_size'],
        lsun_categories=cfg['data']['lsun_categories_train']
    )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg['train']['nworkers'],
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    # Number of labels
    nlabels = min(nlabels, cfg['data']['nlabels'])
    sample_nlabels = min(nlabels, cfg.train.sample_nlabels)

    # Create models
    generator = instantiate(cfg['generator'])
    discriminator = instantiate(cfg['discriminator'])
    print(generator)
    print(discriminator)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optimizer, d_optimizer = build_optimizers(
        generator, discriminator, cfg
    )

    # Use multiple GPUs if possible
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
    )

    # Get model file
    model_file = cfg['train']['model_file']

    # Logger
    logger = Logger(
        log_dir=path.join(cfg.train.out_dir, 'logs'),
        img_dir=path.join(cfg.train.out_dir, 'imgs'),
        monitoring=cfg['train']['monitoring'],
        monitoring_dir=path.join(cfg.train.out_dir, 'monitoring')
    )

    # Distributions
    ydist = get_ydist(nlabels, device=device)
    zdist = get_zdist(cfg['z_dist']['type'], cfg['z_dist']['dim'],
                      device=device)

    # Save for tests
    ntest = cfg.train.batch_size
    x_real, ytest = utils.get_nsamples(train_loader, ntest)
    ytest.clamp_(None, nlabels-1)
    ztest = zdist.sample((ntest,))
    utils.save_images(x_real, path.join(cfg.train.out_dir, 'real.png'))

    # Test generator
    if cfg['train']['take_model_average']:
        generator_test = copy.deepcopy(generator)
        checkpoint_io.register_modules(generator_test=generator_test)
    else:
        generator_test = generator

    # Evaluator
    evaluator = Evaluator(generator_test, zdist, ydist,
                          batch_size=cfg.train.batch_size, device=device)

    # Train
    tstart = t0 = time.time()

    # Load checkpoint if it exists
    try:
        load_dict = checkpoint_io.load(model_file)
    except FileNotFoundError:
        it = epoch_idx = -1
    else:
        it = load_dict.get('it', -1)
        epoch_idx = load_dict.get('epoch_idx', -1)
        logger.load_stats('stats.p')

    # Reinitialize model average if needed
    if (cfg['train']['take_model_average']
            and cfg['train']['model_average_reinit']):
        update_average(generator_test, generator, 0.)

    # Learning rate anneling
    g_scheduler = build_lr_scheduler(g_optimizer, cfg, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, cfg, last_epoch=it)

    # Trainer
    trainer = Trainer(
        generator, discriminator, g_optimizer, d_optimizer,
        gan_type=cfg['train']['gan_type'],
        reg_type=cfg['train']['reg_type'],
        reg_param=cfg['train']['reg_param']
    )

    # Training loop
    print('Start training...')
    while True:
        epoch_idx += 1
        print('Start epoch %d...' % epoch_idx)

        for x_real, y in train_loader:
            it += 1
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']
            logger.add('learning_rates', 'discriminator', d_lr, it=it)
            logger.add('learning_rates', 'generator', g_lr, it=it)

            x_real, y = x_real.to(device), y.to(device)
            y.clamp_(None, nlabels-1)

            # Discriminator updates
            z = zdist.sample((cfg.train.batch_size,))
            dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            if ((it + 1) % cfg.train.d_steps) == 0:
                z = zdist.sample((cfg.train.batch_size,))
                gloss = trainer.generator_trainstep(y, z)
                logger.add('losses', 'generator', gloss, it=it)

                if cfg['train']['take_model_average']:
                    update_average(generator_test, generator,
                                   beta=cfg['train']['model_average_beta'])

            # Print stats
            g_loss_last = logger.get_last('losses', 'generator')
            d_loss_last = logger.get_last('losses', 'discriminator')
            d_reg_last = logger.get_last('losses', 'regularizer')
            print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f'
                  % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last))

            # (i) Sample if necessary
            if (it % cfg['train']['sample_every']) == 0:
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
if __name__ == "__main__":
    main()
