from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from enum import Enum
from utils.utils import make_noise, run_in_background
from utils.prefetch_generator import background


class MaskSynthesizing(Enum):
    LIGHTING = 0
    MEAN_THR = 1


def rgb2gray(img):
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
    return gray


def pair_to_mask(img_orig, img_shifted):
    b_orig, b_shifted = rgb2gray(img_orig), rgb2gray(img_shifted)
    return (b_orig < b_shifted).to(torch.long)


def mean_thr_mask(img_shifted):
    b_shifted = rgb2gray(img_shifted)
    return torch.stack([b > torch.mean(b) for b in b_shifted]).to(torch.long)


def maxes_filter(shifted_images):
    good_samples = []
    r = 3
    device = shifted_images.device
    for im in shifted_images:
        stats = torch.histc(im, 12, -1, 1)
        stats = F.conv1d(stats.view(1, 1, -1), torch.ones([1, 1, r], device=device), padding=r // 2)
        stats = stats.view(-1).cpu().numpy()

        maxes = np.r_[True, stats[1:] >= stats[:-1]] & np.r_[stats[:-1] >= stats[1:], True]
        maxes = np.nonzero(maxes)[0]
        good_samples.append(len(maxes) >= 2)

    return torch.tensor(good_samples)


class MaskGenerator(nn.Module):
    def __init__(self, G, bg_direction, params,
                 mask_preprocessing=(), mask_postprocessing=(),
                 zs=None, z_noise=0.0):
        super(MaskGenerator, self).__init__()
        self.G = G
        self.bg_direction = bg_direction
        self.p = params

        self.mask_preprocessing = mask_preprocessing
        self.mask_postprocessing = mask_postprocessing

        self.zs = zs
        self.z_noise = nn.Parameter(torch.tensor(z_noise, device='cpu'), requires_grad=False)

    @torch.no_grad()
    def make_noise(self, batch_size):
        if self.zs is None:
            return make_noise(batch_size, self.G.dim_z).to(self.z_noise.device)
        else:
            indices = torch.randint(0, len(self.zs), [batch_size], dtype=torch.long)
            z = self.zs[indices].to(self.z_noise.device)
            if self.z_noise > 0.0:
                z = z + self.z_noise * torch.randn_like(z)
            return z


    @torch.no_grad()
    def gen_samples(self, z=None, batch_size=None):
        assert (z is None) ^ (batch_size is None), 'one of: z, batch_size should be provided'

        if z is None:
            z = self.make_noise(batch_size)
        img = self.G(z)
        img_shifted_pos = self.G.gen_shifted(
            z, self.p.latent_shift_r * self.bg_direction.to(z.device))

        if self.p.synthezing == MaskSynthesizing.LIGHTING:
            mask = pair_to_mask(img, img_shifted_pos)

        elif self.p.synthezing == MaskSynthesizing.MEAN_THR:
            mask = mean_thr_mask(img_shifted_pos)

        mask = self._apply_postproc(img, mask)

        return img, img_shifted_pos, mask

    @torch.no_grad()
    def _apply_preproc(self, img, intensity):
        for preproc in self.mask_preprocessing:
            intensity = preproc(img, intensity)
        return intensity

    @torch.no_grad()
    def _apply_postproc(self, img, mask):
        for postproc in self.mask_postprocessing:
            mask = postproc(img, mask)
        return mask

    @torch.no_grad()
    def filter_by_area(self, img_batch, img_pos_batch, ref_batch):
        if self.p.mask_size_up < 1.0:
            ref_size = ref_batch.shape[-2] * ref_batch.shape[-1]
            ref_fraction = ref_batch.sum(dim=[-1, -2]).to(torch.float) / ref_size
            mask = ref_fraction < self.p.mask_size_up
            if torch.all(~mask):
                return None
            img_batch, img_pos_batch, ref_batch = \
                img_batch[mask], img_pos_batch[mask], ref_batch[mask]
        return img_batch, img_pos_batch, ref_batch

    @torch.no_grad()
    def filter_by_maxes_count(self, img_batch, img_pos_batch, ref_batch):
        if self.p.maxes_filter:
            mask = maxes_filter(img_pos_batch)
            if torch.all(~mask):
                return None
            img_batch, img_pos_batch, ref_batch = \
                img_batch[mask], img_pos_batch[mask], ref_batch[mask]
        return img_batch, img_pos_batch, ref_batch

    @torch.no_grad()
    def forward(self, max_retries=100, z=None, return_steps=False):
        img, ref = None, None
        step = 0
        while img is None or img.shape[0] < self.p.batch_size:
            step += 1
            if step > max_retries:
                raise Exception('generator was disable to synthesize mask')

            if z is None or step > 1:
                z = self.make_noise(self.p.batch_size)

            img_batch, img_pos_batch, ref_batch = \
                self.gen_samples(z=z)

            # filtration
            mask_area_filtration = self.filter_by_area(img_batch, img_pos_batch, ref_batch)
            if mask_area_filtration is not None:
                img_batch, img_pos_batch, ref_batch = mask_area_filtration
            else:
                continue

            maxes_count_filtration = self.filter_by_maxes_count(img_batch, img_pos_batch, ref_batch)
            if maxes_count_filtration is not None:
                img_batch, img_pos_batch, ref_batch = maxes_count_filtration
            else:
                continue

            # batch update
            if img is None:
                img, ref = img_batch, ref_batch
            else:
                img = torch.cat([img, img_batch])[:self.p.batch_size]
                ref = torch.cat([ref, ref_batch])[:self.p.batch_size]

        if return_steps:
            return img, ref, step
        return img, ref


@background(max_prefetch=1)
def it_mask_gen(mask_gen, devices, out_device='cpu', delete_orig=True):
    mask_generators = []
    for device in devices:
        mask_generators.append(deepcopy(mask_gen).to(device='cuda:%i' % device))
    if delete_orig:
        del mask_gen

    while True:
        batch_outs_future = [run_in_background(mask_gen_inst) for mask_gen_inst in mask_generators]
        img, ref = \
            map(torch.cat,
                zip(*([f.to(out_device) for f in future.result()] for future in batch_outs_future)))
        yield img.to(out_device), ref.to(out_device)
