import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMAGENET_PATH = '__local_imagenet_train_dir__'


def central_crop(x):
    dims = x.size
    crop = transforms.CenterCrop(min(dims[0], dims[1]))
    return crop(x)


def _id(x):
    return x


def _filename(path):
    return os.path.basename(path).split('.')[0]


def _numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))


class UnannotatedDataset(Dataset):
    def __init__(self, root_dir, numerical_sort=False,
                 transform=transforms.Compose(
                     [
                         transforms.ToTensor(),
                         lambda x: 2 * x - 1
                     ])):
        self.img_files = []
        for root, _, files in os.walk(root_dir):
            for file in _numerical_order(files) if numerical_sort else sorted(files):
                if UnannotatedDataset.file_is_img(file):
                    self.img_files.append(os.path.join(root, file))
        self.transform = transform

    @staticmethod
    def file_is_img(name):
        extension = os.path.basename(name).split('.')[-1]
        return extension in ['jpg', 'jpeg', 'png']

    def align_names(self, target_names):
        new_img_files = []
        img_files_names_dict = {_filename(f): f for f in self.img_files}
        for name in target_names:
            try:
                new_img_files.append(img_files_names_dict[_filename(name)])
            except KeyError:
                print('names mismatch: absent {}'.format(_filename(name)))
        self.img_files = new_img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.img_files[item])
        img = img.convert('RGB')
        if self.transform is not None:
            return self.transform(img)
        else:
            return img


class TransformedDataset(Dataset):
    def __init__(self, source, transform, img_index=0):
        self.source = source
        self.transform = transform
        self.img_index = img_index

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if isinstance(out,tuple):
            return self.transform(out[self.img_index]), out[1 - self.img_index]
        else:
            return self.transform(out)


class LabeledDatasetImagesExtractor(Dataset):
    def __init__(self, ds, img_field=0):
        self.source = ds
        self.img_field = img_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item][self.img_field]


class SegmentationDataset(Dataset):
    def __init__(self, images_root, masks_root, crop=True, size=None, mask_thr=0.5):
        self.mask_thr = mask_thr
        images_ds = UnannotatedDataset(images_root, transform=None)
        masks_ds = UnannotatedDataset(masks_root, transform=None)
        masks_ds.align_names(images_ds.img_files)

        resize = transforms.Compose([
            central_crop if crop else _id,
            transforms.Resize(size) if size is not None else _id,
            transforms.ToTensor()])
        shift_to_zero = lambda x: 2 * x - 1
        self.images_ds = TransformedDataset(images_ds, transforms.Compose([resize, shift_to_zero]))
        self.masks_ds = TransformedDataset(masks_ds, resize)

    def __len__(self):
        return len(self.images_ds)

    def __getitem__(self, index):
        mask = self.masks_ds[index] >= self.mask_thr
        return (self.images_ds[index], mask[0])
