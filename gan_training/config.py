import yaml
from torch import optim
from os import path
from gan_training.models import generator_dict, discriminator_dict
from gan_training.train import toggle_grad
from hydra.utils import instantiate

# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({
            'params': [p],
            'lr': c * base_lr
        })
    return param_groups
