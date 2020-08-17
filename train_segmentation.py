import os
import sys
import argparse
import json
import torch
import numpy as np
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use("Agg")

from UNet.unet_model import UNet
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen
from data import SegmentationDataset
from metrics import model_metrics, IoU, accuracy, F_max
from BigGAN.gan_load import make_big_gan
from postprocessing import connected_components_filter,\
    SegmentationInference, Threshold
from utils.utils import to_image
from visualization import overlayed

DEFAULT_EVAL_KEY = 'id'
THR_EVAL_KEY = 'thr'
SEGMENTATION_RES = 128
BIGBIGAN_WEIGHTS = 'BigGAN/weights/BigBiGAN_x1.pth'
LATENT_DIRECTION = 'BigGAN/weights/bg_direction.pth'


MASK_SYNTHEZ_DICT = {
    'lighting': MaskSynthesizing.LIGHTING,
    'mean_thr': MaskSynthesizing.MEAN_THR,
}


class SegmentationTrainParams(object):
    def __init__(self, **kwargs):
        self.rate = 0.001
        self.steps_per_rate_decay = 8000
        self.rate_decay = 0.2
        self.n_steps = 12000

        self.latent_shift_r = 5.0
        self.batch_size = 95

        self.steps_per_log = 100
        self.steps_per_checkpoint_save = 1000
        self.steps_per_validation = 1000
        self.test_samples_count = 1000

        self.synthezing = MaskSynthesizing.LIGHTING
        self.mask_size_up = 0.5
        self.connected_components = True
        self.maxes_filter = True

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val

        if isinstance(self.synthezing, str):
            self.synthezing = MASK_SYNTHEZ_DICT[self.synthezing]


def main():
    parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
    parser.add_argument('--args', type=str, default=None, help='json with all arguments')

    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gan_weights', type=str, default=BIGBIGAN_WEIGHTS)
    parser.add_argument('--bg_direction', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gen_devices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--z', type=str, default=None)
    parser.add_argument('--z_noise', type=float, default=0.0)
    parser.add_argument('--val_images_dirs', nargs='*', type=str, default=[None])
    parser.add_argument('--val_masks_dirs', nargs='*', type=str, default=[None])

    for key, val in SegmentationTrainParams().__dict__.items():
        val_type = type(val) if key is not 'synthezing' else str
        parser.add_argument('--{}'.format(key), type=val_type, default=None)

    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    if args.args is not None:
        with open(args.args) as args_json:
            args_dict = json.load(args_json)
            args.__dict__.update(**args_dict)
    if (args.z is not None) and (not os.path.isfile(args.z)):
        print('No valid z-embeddings file is provided.')
        args.z = None

    # save run p
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    G = make_big_gan(args.gan_weights).eval().cuda()
    bg_direction = torch.load(args.bg_direction)

    model = UNet().train().cuda()
    train_params = SegmentationTrainParams(**args.__dict__)
    print('run train with p: {}'.format(train_params.__dict__))

    synthetic_score = train_segmentation(
        G, bg_direction, model, train_params, args.out,
        args.gen_devices, val_dirs=[args.val_images_dirs[0], args.val_masks_dirs[0]],
        zs=args.z, z_noise=args.z_noise)

    score_json = os.path.join(args.out, 'score.json')
    update_out_json({'synthetic': synthetic_score}, score_json)
    print('Synthetic data score: {}'.format(synthetic_score))

    if len(args.val_images_dirs) > 0:
        evaluate_all_wrappers(model, score_json,
                              args.val_images_dirs, args.val_masks_dirs)


def gen_postprocessing(params):
    postprocessing = []
    if params.connected_components:
        postprocessing.append(connected_components_filter)
    return postprocessing


def train_segmentation(G, bg_direction, model, params, out_dir,
                       gen_devices, val_dirs=None, zs=None, z_noise=0.0):
    model.train()
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'))

    params.batch_size = params.batch_size // len(gen_devices)

    if zs is not None and os.path.isfile(zs):
        zs = torch.from_numpy(np.load(zs))
    mask_postprocessing = gen_postprocessing(params)
    mask_generator = MaskGenerator(
        G, bg_direction, params, [], mask_postprocessing,
        zs=zs, z_noise=z_noise).cuda().eval()
    # form test batch
    num_test_steps = params.test_samples_count // params.batch_size
    test_samples = [mask_generator() for _ in range(num_test_steps)]
    test_samples = [[s[0].cpu(), s[1].cpu()] for s in test_samples]

    optimizer = torch.optim.Adam(model.parameters(), lr=params.rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)
    criterion = torch.nn.CrossEntropyLoss()

    start_step = 0
    checkpoint = os.path.join(out_dir, 'checkpoint.pth')
    if os.path.isfile(checkpoint):
        start_step = load_checkpoint(model, optimizer, lr_scheduler, checkpoint)
        print('Starting from step {} checkpoint'.format(start_step))

    print('start loop', flush=True)
    for step, (img, ref) in enumerate(it_mask_gen(mask_generator, gen_devices, 'cpu')):
        step += start_step
        model.zero_grad()
        prediction = model(img.cuda())
        loss = criterion(prediction, ref.cuda())

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if step > 0 and step % params.steps_per_checkpoint_save == 0:
            print('Step {}: saving checkpoint'.format(step))
            save_checkpoint(model, optimizer, lr_scheduler, step, checkpoint)

        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), step)

        if step > 0 and step % params.steps_per_log == 0:
            with torch.no_grad():
                loss = 0.0
                for img, ref in test_samples:
                    prediction = model(img.cuda())
                    loss += criterion(prediction, ref.cuda()).item()
            loss = loss / num_test_steps
            print('{}% | step {}: {}'.format(
                int(100.0 * step / params.n_steps), step, loss))
            writer.add_scalar('val/loss', loss, step)

        is_val_step = \
            (step > 0 and step % params.steps_per_validation == 0) or (step == params.n_steps)
        if is_val_step and val_dirs is not None:
            print('Step: {} | evaluation'.format(step))
            model.eval()
            eval_dict = evaluate(
                SegmentationInference(model, resize_to=SEGMENTATION_RES),
                val_dirs[0], val_dirs[1], (F_max,))
            update_out_json(eval_dict, os.path.join(out_dir, 'score.json'))
            model.train()
        if step == 0:
            to_image(overlayed(img[:16], ref[:16].unsqueeze(1)), True).save(f'{out_dir}/gen_sample.png')
        if step == params.n_steps:
            break

    model.eval()
    torch.save(model.state_dict(), os.path.join(out_dir, 'segmentation.pth'))

    return evaluate_gan_mask_generator(
        model, G, bg_direction, params, mask_postprocessing, zs, z_noise, params.test_samples_count)


def evaluate_gan_mask_generator(model, G, bg_direction, params,
                                mask_postprocessing, zs, z_noise, num_steps):
    mask_generator = MaskGenerator(
        G, bg_direction, params, [], mask_postprocessing,
        zs=zs, z_noise=z_noise).cuda().eval()
    def it():
        while True:
            sample = mask_generator()
            for img, mask in zip(sample[0], sample[1]):
                yield img.unsqueeze(0), mask

    score = {
        DEFAULT_EVAL_KEY:
            model_metrics(SegmentationInference(model), it(), num_steps, (F_max,)),
        THR_EVAL_KEY:
            model_metrics(Threshold(model), it(), num_steps, (IoU, accuracy)),
    }

    return score


def save_checkpoint(model, opt, scheduler, step, checkpoint):
    state_dict = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step}
    torch.save(state_dict, checkpoint)


def load_checkpoint(model, opt, scheduler, checkpoint):
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    opt.load_state_dict(state_dict['opt'])
    scheduler.load_state_dict(state_dict['scheduler'])
    return state_dict['step']


def update_out_json(eval_dict, out_json):
    out_dict = {}
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            out_dict = json.load(f)

    with open(out_json, 'w') as out:
        out_dict.update(eval_dict)
        json.dump(out_dict, out)


@torch.no_grad()
def evaluate(segmentation_model, images_dir, masks_dir, metrics, size=None):
    segmentation_dl = torch.utils.data.DataLoader(
        SegmentationDataset(images_dir, masks_dir, size=size, crop=False), 1, shuffle=False)

    eval_out = model_metrics(segmentation_model, segmentation_dl, stats=metrics)
    print('Segmenation model', eval_out)
    return eval_out


def keys_in_dict_tree(dict_tree, keys):
    for key in keys:
        if key not in dict_tree.keys():
            return False
        dict_tree = dict_tree[key]
    return True


@torch.no_grad()
def evaluate_all_wrappers(model, out_file, val_images_dirs, val_masks_dirs):
    model.eval()
    evaluation_dict = {}
    if os.path.isfile(out_file):
        with open(out_file, 'r') as f:
            evaluation_dict = json.load(f)

    for val_imgs, val_dirs in zip(val_images_dirs, val_masks_dirs):
        ds_name = val_imgs.split('/')[-2]
        print('\nEvaluating {}'.format(ds_name))
        if ds_name not in evaluation_dict.keys():
            evaluation_dict[ds_name] = {}

        if not keys_in_dict_tree(evaluation_dict, [ds_name, DEFAULT_EVAL_KEY]):
            evaluation_dict[ds_name][DEFAULT_EVAL_KEY] = evaluate(
                SegmentationInference(model, resize_to=SEGMENTATION_RES),
                val_imgs, val_dirs, (F_max,))
            update_out_json(evaluation_dict, out_file)

        if not keys_in_dict_tree(evaluation_dict, [ds_name, THR_EVAL_KEY]):
            evaluation_dict[ds_name][THR_EVAL_KEY] = evaluate(
                Threshold(model, resize_to=SEGMENTATION_RES), val_imgs, val_dirs, (IoU, accuracy))
            update_out_json(evaluation_dict, out_file)


if __name__ == '__main__':
    main()
