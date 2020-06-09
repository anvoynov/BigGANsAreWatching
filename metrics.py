import numpy as np
import torch

from postprocessing import resize


def IoU(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def accuracy(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    return F.mean(dim=0).max().item()


@torch.no_grad()
def model_metrics(segmetation_model, dataloder, n_steps=None,
                  stats=(IoU, accuracy, F_max), prob_bins=255):
    avg_values = {}
    precisions = []
    recalls = []
    out_dict = {}

    n_steps = len(dataloder) if n_steps is None else n_steps
    step = 0
    for step, (img, mask) in enumerate(dataloder):
        img, mask = img.cuda(), mask.cuda()

        if img.shape[-2:] != mask.shape[-2:]:
            mask = resize(mask, img.shape[-2:])

        prediction = segmetation_model(img)

        for metric in stats:
            method = metric.__name__
            if method not in avg_values and metric != F_max:
                avg_values[method] = 0.0

            if metric != F_max:
                avg_values[method] += metric(mask, prediction)
            else:
                p, r = [], []
                splits = 2.0 * prediction.mean(dim=0) if prob_bins is None else \
                    np.arange(0.0, 1.0, 1.0 / prob_bins)

                for split in splits:
                    pr = precision_recall(mask, prediction > split)
                    p.append(pr[0])
                    r.append(pr[1])
                precisions.append(p)
                recalls.append(r)

        step += 1
        if n_steps is not None and step >= n_steps:
            break

    for metric in stats:
        method = metric.__name__
        if metric == F_max:
            out_dict[method] = F_max(torch.tensor(precisions), torch.tensor(recalls))
        else:
            out_dict[method] = avg_values[method] / step

    return out_dict
