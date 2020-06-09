import torch
from torch import nn
from BigGAN.model import BigGAN
from BigGAN.gan_with_shift import gan_with_shift


class UnconditionalBigGAN(nn.Module):
    def __init__(self, big_gan):
        super(UnconditionalBigGAN, self).__init__()
        self.big_gan = big_gan
        self.dim_z = self.big_gan.dim_z

    def forward(self, z):
        classes = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
        return self.big_gan(z, self.big_gan.shared(classes))


def make_biggan_config(resolution):
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}
    config = {
        'G_param': 'SN', 'D_param': 'SN',
        'G_ch': 96, 'D_ch': 96,
        'D_wide': True, 'G_shared': True,
        'shared_dim': 128, 'dim_z': dim_z_dict[resolution],
        'hier': True, 'cross_replica': False,
        'mybn': False, 'G_activation': nn.ReLU(inplace=True),
        'G_attn': attn_dict[resolution],
        'norm_style': 'bn',
        'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
        'G_fp16': False, 'G_mixed_precision': False,
        'accumulate_stats': False, 'num_standing_accumulations': 16,
        'G_eval_mode': True,
        'BN_eps': 1e-04, 'SN_eps': 1e-04,
        'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': resolution,
        'n_classes': 1000}
    return config


@gan_with_shift
def make_big_gan(weights_root, resolution=128):
    config = make_biggan_config(resolution)
    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(weights_root, map_location=torch.device('cpu')), strict=False)

    return UnconditionalBigGAN(G).cuda()
