"""Utilities for converting TFHub BigGAN generator weights to PyTorch.

Recommended usage:

To convert all BigGAN variants and generate test samples, use:

```bash
CUDA_VISIBLE_DEVICES=0 python converter.py --generate_samples
```

See `parse_args` for additional options.
"""

import os
import h5py
import torch
import tensorflow as tf
import parse

from tf_tools import proxy_biggan
from BigGAN.model.BigGAN import Generator
from BigGAN.gan_load import make_biggan_config

import io
DEVICE = 'cuda'


def dump_tfhub_to_hdf5(sess):
    buffer = io.BytesIO()
    h5f = h5py.File(buffer, 'w')

    vars = tf.compat.v1.global_variables()
    names = [v.name for v in vars]
    values = sess.run(vars)

    for name, val in zip(names, values):
        h5f.create_dataset(name, data=val)
    h5f.close()

    return h5py.File(buffer, 'r')


class TFHub2Proxy(object):
    TF_ROOT = 'module'

    NUM_GBLOCK = {
        128: 5,
        256: 6,
        512: 7
    }

    w = 'w'
    b = 'b'
    u = 'u0'
    v = 'u1'
    gamma = 'gamma'
    beta = 'beta'

    def __init__(self, state_dict, tf_weights, resolution=256, load_ema=True, verbose=False):
        self.state_dict = state_dict
        self.tf_weights = tf_weights
        self.resolution = resolution
        self.verbose = verbose
        if load_ema:
            for name in ['w', 'b', 'gamma', 'beta']:
                setattr(self, name, getattr(self, name) + '/ema_b999900')

    def load(self):
        self.load_generator()
        return self.state_dict

    def load_generator(self):
        GENERATOR_ROOT = os.path.join(self.TF_ROOT, 'Generator')

        for i in range(self.NUM_GBLOCK[self.resolution]):
            name_tf = os.path.join(GENERATOR_ROOT, 'GBlock')
            name_tf += '_{}'.format(i) if i != 0 else ''
            self.load_GBlock('GBlock.{}.'.format(i), name_tf)

        self.load_attention('attention.', os.path.join(GENERATOR_ROOT, 'attention'))
        self.load_linear('linear', os.path.join(self.TF_ROOT, 'linear'), bias=False)
        self.load_snlinear('G_linear', os.path.join(GENERATOR_ROOT, 'G_Z', 'G_linear'))
        self.load_colorize('colorize', os.path.join(GENERATOR_ROOT, 'conv_2d'))
        self.load_ScaledCrossReplicaBNs('ScaledCrossReplicaBN',
                                        os.path.join(GENERATOR_ROOT, 'ScaledCrossReplicaBN'))

    def load_linear(self, name_pth, name_tf, bias=True):
        self.state_dict[name_pth + '.weight'] = self.load_tf_tensor(name_tf, self.w).permute(1, 0)
        if bias:
            self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_snlinear(self, name_pth, name_tf, bias=True):
        self.state_dict[name_pth + '.module.weight_u'] = self.load_tf_tensor(name_tf,
                                                                             self.u).squeeze()
        self.state_dict[name_pth + '.module.weight_v'] = self.load_tf_tensor(name_tf,
                                                                             self.v).squeeze()
        self.state_dict[name_pth + '.module.weight_bar'] = self.load_tf_tensor(name_tf,
                                                                               self.w).permute(1, 0)
        if bias:
            self.state_dict[name_pth + '.module.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_colorize(self, name_pth, name_tf):
        self.load_snconv(name_pth, name_tf)

    def load_GBlock(self, name_pth, name_tf):
        self.load_convs(name_pth, name_tf)
        self.load_HyperBNs(name_pth, name_tf)

    def load_convs(self, name_pth, name_tf):
        self.load_snconv(name_pth + 'conv0', os.path.join(name_tf, 'conv0'))
        self.load_snconv(name_pth + 'conv1', os.path.join(name_tf, 'conv1'))
        self.load_snconv(name_pth + 'conv_sc', os.path.join(name_tf, 'conv_sc'))

    def load_snconv(self, name_pth, name_tf, bias=True):
        self.state_dict[name_pth + '.module.weight_u'] = self.load_tf_tensor(name_tf,
                                                                             self.u).squeeze()
        self.state_dict[name_pth + '.module.weight_v'] = self.load_tf_tensor(name_tf,
                                                                             self.v).squeeze()
        self.state_dict[name_pth + '.module.weight_bar'] = self.load_tf_tensor(name_tf,
                                                                               self.w).permute(3, 2,
                                                                                               0, 1)
        if bias:
            self.state_dict[name_pth + '.module.bias'] = self.load_tf_tensor(name_tf,
                                                                             self.b).squeeze()

    def load_conv(self, name_pth, name_tf, bias=True):

        self.state_dict[name_pth + '.weight_u'] = self.load_tf_tensor(name_tf, self.u).squeeze()
        self.state_dict[name_pth + '.weight_v'] = self.load_tf_tensor(name_tf, self.v).squeeze()
        self.state_dict[name_pth + '.weight_bar'] = self.load_tf_tensor(name_tf, self.w).permute(3,
                                                                                                 2,
                                                                                                 0,
                                                                                                 1)
        if bias:
            self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.b)

    def load_HyperBNs(self, name_pth, name_tf):
        self.load_HyperBN(name_pth + 'HyperBN', os.path.join(name_tf, 'HyperBN'))
        self.load_HyperBN(name_pth + 'HyperBN_1', os.path.join(name_tf, 'HyperBN_1'))

    def load_ScaledCrossReplicaBNs(self, name_pth, name_tf):
        self.state_dict[name_pth + '.bias'] = self.load_tf_tensor(name_tf, self.beta).squeeze()
        self.state_dict[name_pth + '.weight'] = self.load_tf_tensor(name_tf, self.gamma).squeeze()
        self.state_dict[name_pth + '.running_mean'] = self.load_tf_tensor(name_tf + 'bn',
                                                                          'accumulated_mean')
        self.state_dict[name_pth + '.running_var'] = self.load_tf_tensor(name_tf + 'bn',
                                                                         'accumulated_var')
        self.state_dict[name_pth + '.num_batches_tracked'] = torch.tensor(
            self.tf_weights[os.path.join(name_tf + 'bn', 'accumulation_counter:0')][()],
            dtype=torch.float32)

    def load_HyperBN(self, name_pth, name_tf):
        beta = name_pth + '.beta_embed.module'
        gamma = name_pth + '.gamma_embed.module'
        self.state_dict[beta + '.weight_u'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'),
                                                                  self.u).squeeze()
        self.state_dict[gamma + '.weight_u'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'),
                                                                   self.u).squeeze()
        self.state_dict[beta + '.weight_v'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'),
                                                                  self.v).squeeze()
        self.state_dict[gamma + '.weight_v'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'),
                                                                   self.v).squeeze()
        self.state_dict[beta + '.weight_bar'] = self.load_tf_tensor(os.path.join(name_tf, 'beta'),
                                                                    self.w).permute(1, 0)
        self.state_dict[gamma +
                        '.weight_bar'] = self.load_tf_tensor(os.path.join(name_tf, 'gamma'),
                                                             self.w).permute(1, 0)

        cr_bn_name = name_tf.replace('HyperBN', 'CrossReplicaBN')
        self.state_dict[name_pth + '.bn.running_mean'] = self.load_tf_tensor(cr_bn_name,
                                                                             'accumulated_mean')
        self.state_dict[name_pth + '.bn.running_var'] = self.load_tf_tensor(cr_bn_name,
                                                                            'accumulated_var')
        self.state_dict[name_pth + '.bn.num_batches_tracked'] = torch.tensor(
            self.tf_weights[os.path.join(cr_bn_name, 'accumulation_counter:0')][()],
            dtype=torch.float32)

    def load_attention(self, name_pth, name_tf):

        self.load_snconv(name_pth + 'theta', os.path.join(name_tf, 'theta'), bias=False)
        self.load_snconv(name_pth + 'phi', os.path.join(name_tf, 'phi'), bias=False)
        self.load_snconv(name_pth + 'g', os.path.join(name_tf, 'g'), bias=False)
        self.load_snconv(name_pth + 'o_conv', os.path.join(name_tf, 'o_conv'), bias=False)
        self.state_dict[name_pth + 'gamma'] = self.load_tf_tensor(name_tf, self.gamma)

    def load_tf_tensor(self, prefix, var, device='0'):
        name = os.path.join(prefix, var) + ':{}'.format(device)
        return torch.from_numpy(self.tf_weights[name][:])


def proxy_to_torch(hub_dict, resolution=128):
    weightname_dict = {'weight_u': 'u0', 'weight_bar': 'weight', 'bias': 'bias'}
    convnum_dict = {'conv0': 'conv1', 'conv1': 'conv2', 'conv_sc': 'conv_sc'}
    attention_blocknum = {128: 3, 256: 4, 512: 3}[resolution]
    proxy_to_torch_names = {'linear.weight': 'shared.weight',  # This is actually the shared weight
              # Linear stuff
              'G_linear.module.weight_bar': 'linear.weight',
              'G_linear.module.bias': 'linear.bias',
              'G_linear.module.weight_u': 'linear.u0',
              # output layer stuff
              'ScaledCrossReplicaBN.weight': 'output_layer.0.gain',
              'ScaledCrossReplicaBN.bias': 'output_layer.0.bias',
              'ScaledCrossReplicaBN.running_mean': 'output_layer.0.stored_mean',
              'ScaledCrossReplicaBN.running_var': 'output_layer.0.stored_var',
              'colorize.module.weight_bar': 'output_layer.2.weight',
              'colorize.module.bias': 'output_layer.2.bias',
              'colorize.module.weight_u': 'output_layer.2.u0',
              # Attention stuff
              'attention.gamma': 'blocks.%d.1.gamma' % attention_blocknum,
              'attention.theta.module.weight_u': 'blocks.%d.1.theta.u0' % attention_blocknum,
              'attention.theta.module.weight_bar': 'blocks.%d.1.theta.weight' % attention_blocknum,
              'attention.phi.module.weight_u': 'blocks.%d.1.phi.u0' % attention_blocknum,
              'attention.phi.module.weight_bar': 'blocks.%d.1.phi.weight' % attention_blocknum,
              'attention.g.module.weight_u': 'blocks.%d.1.g.u0' % attention_blocknum,
              'attention.g.module.weight_bar': 'blocks.%d.1.g.weight' % attention_blocknum,
              'attention.o_conv.module.weight_u': 'blocks.%d.1.o.u0' % attention_blocknum,
              'attention.o_conv.module.weight_bar': 'blocks.%d.1.o.weight' % attention_blocknum,
              }

    # Loop over the hub dict and build the proxy_to_torch_names map
    for name in hub_dict.keys():
        if 'GBlock' in name:
            if 'HyperBN' not in name:  # it's a conv
                out = parse.parse('GBlock.{:d}.{}.module.{}', name)
                blocknum, convnum, weightname = out
                if weightname not in weightname_dict:
                    continue  # else hyperBN in
                out_name = 'blocks.%d.0.%s.%s' % (blocknum, convnum_dict[convnum], weightname_dict[
                    weightname])  # Increment conv number by 1
            else:  # hyperbn not conv
                BNnum = 2 if 'HyperBN_1' in name else 1
                if 'embed' in name:
                    out = parse.parse('GBlock.{:d}.{}.module.{}', name)
                    blocknum, gamma_or_beta, weightname = out
                    if weightname not in weightname_dict:  # Ignore weight_v
                        continue
                    out_name = 'blocks.%d.0.bn%d.%s.%s' % (
                    blocknum, BNnum, 'gain' if 'gamma' in gamma_or_beta else 'bias',
                    weightname_dict[weightname])
                else:
                    out = parse.parse('GBlock.{:d}.{}.bn.{}', name)
                    blocknum, dummy, mean_or_var = out
                    if 'num_batches_tracked' in mean_or_var:
                        continue
                    out_name = 'blocks.%d.0.bn%d.%s' % (
                    blocknum, BNnum, 'stored_mean' if 'mean' in mean_or_var else 'stored_var')
            proxy_to_torch_names[name] = out_name

    # Invert the proxy_to_torch_names map
    torch_to_proxy_names = {proxy_to_torch_names[item]: item for item in proxy_to_torch_names}
    new_dict = {}
    dimz_dict = {128: 20, 256: 20, 512: 16}
    for item in torch_to_proxy_names:
        # Swap input dim ordering on batchnorm bois to account for my arbitrary change of ordering when concatenating Ys and Zs
        if ('bn' in item and 'weight' in item) and ('gain' in item or 'bias' in item) and (
                'output_layer' not in item):
            new_dict[item] = torch.cat([hub_dict[torch_to_proxy_names[item]][:, -128:],
                                        hub_dict[torch_to_proxy_names[item]][:, :dimz_dict[resolution]]], 1)
        # Reshape the first linear weight, bias, and u0
        elif item == 'linear.weight':
            new_dict[item] = hub_dict[torch_to_proxy_names[item]]\
                .contiguous()\
                .view(4, 4, 96 * 16, -1)\
                .permute(2, 0, 1, 3)\
                .contiguous()\
                .view(-1, dimz_dict[resolution])
        elif item == 'linear.bias':
            new_dict[item] = hub_dict[torch_to_proxy_names[item]]\
                .view(4, 4, 96 * 16)\
                .permute(2, 0, 1)\
                .contiguous()\
                .view(-1)
        elif item == 'linear.u0':
            new_dict[item] = hub_dict[torch_to_proxy_names[item]]\
                .view(4, 4, 96 * 16)\
                .permute(2, 0, 1)\
                .contiguous()\
                .view(1, -1)
        elif torch_to_proxy_names[item] == 'linear.weight':  # THIS IS THE SHARED WEIGHT NOT THE FIRST LINEAR LAYER
            # Transpose shared weight so that it's an embedding
            new_dict[item] = hub_dict[torch_to_proxy_names[item]].t()
        elif 'weight_u' in torch_to_proxy_names[item]:  # Unsqueeze u0s
            new_dict[item] = hub_dict[torch_to_proxy_names[item]].unsqueeze(0)
        else:
            new_dict[item] = hub_dict[torch_to_proxy_names[item]]
    return new_dict


def convert_tf_to_torch(resolution, sess=None, no_ema=False, verbose=False):
    tf_weights = dump_tfhub_to_hdf5(sess)

    G_temp = getattr(proxy_biggan, 'Generator{}'.format(resolution))()
    state_dict_temp = G_temp.state_dict()

    converter = TFHub2Proxy(state_dict_temp, tf_weights, resolution=resolution,
                            load_ema=(not no_ema), verbose=verbose)

    state_dict_v1 = converter.load()
    state_dict = proxy_to_torch(state_dict_v1, resolution)
    # Get the config, build the model
    config = make_biggan_config(resolution)
    G = Generator(**config)
    G.load_state_dict(state_dict, strict=False)  # Ignore missing sv0 entries

    return G
