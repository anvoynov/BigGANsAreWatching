from functools import partial
from tqdm.auto import trange
import numpy as np
import torch
from torch import nn


class BestOfTransformator(nn.Module):
    def __init__(self, transform_factory, n_transforms=2):
        super(BestOfTransformator, self).__init__()
        
        self.transforms = nn.ModuleList([
                transform_factory() for _ in range(n_transforms)
        ])
        for t in self.transforms:
            t.weight.data = 0.01 * t.weight.data + torch.eye(t.weight.data.shape[0])
        self.n_transforms = n_transforms

    def forward(self, x, target):
        xs = torch.stack([torch.clamp(t(x), -1.0, 1.0) for t in self.transforms])

        dists = torch.norm(xs - target.unsqueeze(0).repeat(self.n_transforms, 1, 1), dim=-1)
        best_index = torch.argmin(dists, dim=0)

        out = xs[best_index, torch.arange(xs.shape[1], device=xs.device)]
        return out, best_index


def set_requires_grad(module, requires_grad=True):
    for p in module.parameters():
        p.requires_grad = requires_grad


def to_flatten_batch(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x.permute(1, 0, 2, 3).reshape([3, -1]).T


def from_flatten_batch(x, size, n_channel=3):
    return x.to(torch.float).T.reshape([n_channel, -1, size, size]).permute(1, 0, 2, 3)


def operators_clasterization(G, h, zs, zs_test=None,
                             verbose=True, n_classes=2, train_steps=200, test_steps=5,
                             batch=4, rate=0.005, steps_per_batch=1):
    set_requires_grad(G, False)
    if zs_test is None:
        zs_test = zs

    best_of_transformator = BestOfTransformator(
        partial(nn.Linear, in_features=3, out_features=3, bias=True), n_classes).cuda()

    opt = torch.optim.Adam(best_of_transformator.parameters(), lr=rate)
    losses = []

    for i in trange(train_steps + test_steps) if verbose else range(train_steps + test_steps):
        if i % steps_per_batch == 0 or i >= train_steps:
            with torch.no_grad():
                if i < train_steps:
                    indices = torch.randint(0, len(zs), batch)
                    z = zs[indices].cuda()
                else:
                    i_test = i - test_steps
                    indices = list(range(batch * i_test, batch * (i_test + 1)))
                    z = zs_test[indices].cuda()

                img_orig = G(z)
                img_shifted = G.gen_shifted(z, h)

        x = to_flatten_batch(img_orig)
        target = to_flatten_batch(img_shifted)
        out, mask = best_of_transformator(x, target)

        diff = (out - target).norm(dim=1)
        loss = diff.mean()
        if i % 50 == 0 and verbose:
            print(f'{loss.item():0.2f}')

        # join train and eval in a single loop
        if i < train_steps:
            loss.backward()
            opt.step()
            best_of_transformator.zero_grad()
        else:
            losses.append(loss.item())

    size = img_orig.shape[-1]
    mask = from_flatten_batch(mask, size, 1)
    out = from_flatten_batch(out, size, 3)
    diff = from_flatten_batch(mask, size, 1)
    print(np.mean(losses))

    return mask, img_orig, img_shifted, out, diff, best_of_transformator.transforms, np.mean(losses)
