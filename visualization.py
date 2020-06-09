import torch
from matplotlib import pyplot as plt
from utils.utils import to_image
from torchvision.utils import make_grid


def overlayed(img, mask, channel=1, nrow=1):
    imgs_grid = make_grid(torch.clamp(img, -1, 1), nrow=nrow, padding=5)
    masks_grid = make_grid(mask, nrow=nrow, padding=5, pad_value=0.5).cpu()[0]

    overlay = 0.5 * (1 + imgs_grid.clone().cpu())
    overlay -= 0.5 * masks_grid.unsqueeze(0)
    overlay[channel] += masks_grid

    return torch.clamp(overlay, 0, 1)


def draw_with_mask(img, masks, names=None, horizontal=True):
    if horizontal:
        fig, axs = plt.subplots(ncols=len(masks) + 1, figsize=(3 * len(masks) + 3, 3), dpi=250)
    else:
        fig, axs = plt.subplots(nrows=len(masks) + 1, figsize=(len(img), len(masks) + 3), dpi=250)
    nrow = 1 if horizontal else img.shape[0]

    axs[0].axis('off')
    axs[0].set_title('original', fontsize=8)
    axs[0].imshow(to_image(make_grid(torch.clamp(img, -1, 1), nrow=nrow, padding=5)))

    for i, mask in enumerate(masks):
        overlay = overlayed(img, mask, int(i > 0), nrow)
        ax = axs[i + 1]
        ax.axis('off')
        if names is not None:
            ax.set_title(names[i], fontsize=8)
        ax.imshow(to_image(overlay, True))

    return fig
