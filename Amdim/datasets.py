import os
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms


INTERP = 3


class Dataset(Enum):
    C10 = 1
    C100 = 2
    STL10 = 3
    IN128 = 4
    PLACES205 = 5
    QA10SCENES = 6
    EMCAS = 7


def get_encoder_size(dataset):
    if dataset in [Dataset.C10, Dataset.C100]:
        return 32
    if dataset == Dataset.STL10:
        return 64
    if dataset in [Dataset.IN128, Dataset.PLACES205]:
        return 128
    if dataset == Dataset.QA10SCENES:
        return 128
    if dataset in [Dataset.EMCAS]:
        return 128
    raise RuntimeError("Couldn't get encoder size, unknown dataset: {}".format(dataset))


def get_dataset(dataset_name):
    try:
        return Dataset[dataset_name.upper()]
    except KeyError as e:
        raise KeyError("Unknown dataset '" + dataset_name + "'. Must be one of "
                       + ', '.join([d.name for d in Dataset]))


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsSTL10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        # image augmentation functions
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        rand_crop = \
            transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)

        self.test_transform = transforms.Compose([
            transforms.Resize(70, interpolation=INTERP),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize
        ])

        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsImageNet128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=INTERP),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2


class TransformsQA10Scenes:
    '''
    QA 10 Scenes dataset, for use with 224x224 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        # self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # self.flip_tb = transforms.RandomVerticalFlip(p=0.5)
        # self.rotation = transforms.RandomRotation(degrees=20, resample=Image.BILINEAR) # fill_mode is not reflect yet

        post_transform = transforms.Compose([
            transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ])
        self.test_transform = transforms.Compose([ # When testing, don't need to flip or rotate
            transforms.Resize(128, interpolation=Image.NEAREST),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            transforms.Resize(128, interpolation=Image.NEAREST),
            post_transform
        ])

    def __call__(self, inp):
        # inp = self.flip_lr(inp)
        # inp = self.flip_tb(inp)
        # inp = self.rotation(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2

class TransformsEmcas:
    '''
    QA 10 Scenes dataset, for use with 224x224 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        self.flip_tb = transforms.RandomVerticalFlip(p=0.5)
        self.rotation = transforms.RandomRotation(degrees=20, resample=Image.BILINEAR, fill=0) # fill_mode is not reflect yet

        post_transform = transforms.Compose([
            transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ])
        self.test_transform = transforms.Compose([ # When testing, don't need to flip or rotate
            transforms.Resize(128, interpolation=Image.NEAREST),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            transforms.Resize(128, interpolation=Image.NEAREST),
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                        interpolation=INTERP),
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        inp = self.flip_tb(inp)
        inp = self.rotation(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        # print(out1.shape)
        # print(out2.shape)
        return out1, out2


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def build_dataset(dataset, batch_size, input_dir=None, labeled_only=False, n_classes=8, train_scene='Scene1', test_scene='Scene2'):

    train_dir, val_dir = _get_directories(dataset, input_dir, train_scene, test_scene)

    if dataset == Dataset.C10:
        num_classes = 10
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        train_dataset = datasets.CIFAR10(root='/tmp/data/',
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root='/tmp/data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif dataset == Dataset.C100:
        num_classes = 100
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        train_dataset = datasets.CIFAR100(root='/tmp/data/',
                                          train=True,
                                          transform=train_transform,
                                          download=True)
        test_dataset = datasets.CIFAR100(root='/tmp/data/',
                                         train=False,
                                         transform=test_transform,
                                         download=True)
    elif dataset == Dataset.STL10:
        num_classes = 10
        train_transform = TransformsSTL10()
        test_transform = train_transform.test_transform
        train_split = 'train' if labeled_only else 'train+unlabeled'
        train_dataset = datasets.STL10(root='/tmp/data/',
                                       split=train_split,
                                       transform=train_transform,
                                       download=True)
        test_dataset = datasets.STL10(root='/tmp/data/',
                                      split='test',
                                      transform=test_transform,
                                      download=True)
    elif dataset == Dataset.IN128:
        num_classes = 1000
        train_transform = TransformsImageNet128()
        test_transform = train_transform.test_transform
        train_dataset = ImageFolderWithPaths(train_dir, train_transform)
        test_dataset = ImageFolderWithPaths(val_dir, test_transform)
    elif dataset == Dataset.PLACES205:
        num_classes = 1000
        train_transform = TransformsImageNet128()
        test_transform = train_transform.test_transform
        train_dataset = ImageFolderWithPaths(train_dir, train_transform)
        test_dataset = ImageFolderWithPaths(val_dir, test_transform)
    elif dataset == Dataset.QA10SCENES:
        num_classes = n_classes
        train_transform = TransformsQA10Scenes()
        test_transform = train_transform.test_transform # When testing, don't need to flip or rotate
        train_dataset = ImageFolderWithPaths(train_dir, train_transform)
        test_dataset = ImageFolderWithPaths(val_dir, test_transform)
    elif dataset == Dataset.EMCAS:
        num_classes = n_classes
        train_transform = TransformsEmcas()
        test_transform = train_transform.test_transform
        # train_dataset = datasets.ImageFolder(train_dir, train_transform)
        # test_dataset = datasets.ImageFolder(val_dir, test_transform)
        train_dataset = ImageFolderWithPaths(train_dir, train_transform)
        test_dataset = ImageFolderWithPaths(val_dir, test_transform)

    # build pytorch dataloaders for the datasets
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=4)
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=False,
                                    num_workers=4)

    return train_loader, test_loader, num_classes


def _get_directories(dataset, input_dir, train_scene, test_scene):
    if dataset in [Dataset.C10, Dataset.C100, Dataset.STL10]:
        # Pytorch will download those datasets automatically
        return None, None
    if dataset == Dataset.IN128:
        train_dir = os.path.join(input_dir, 'ILSVRC2012_img_train/')
        val_dir = os.path.join(input_dir, 'ILSVRC2012_img_val/')
    elif dataset == Dataset.PLACES205:
        train_dir = os.path.join(input_dir, 'places205_256_train/')
        val_dir = os.path.join(input_dir, 'places205_256_val/')
    elif dataset == Dataset.QA10SCENES:
        # train_dir = os.path.join(input_dir, 'Scene1/')
        # val_dir = os.path.join(input_dir, 'Scene2/')
        train_dir = os.path.join(input_dir, "".join([train_scene, '/']))
        val_dir = os.path.join(input_dir, "".join([test_scene, '/']))
    elif dataset == Dataset.EMCAS:
        train_dir = os.path.join(input_dir, "".join([train_scene, '/']))
        val_dir = os.path.join(input_dir, "".join([test_scene, '/']))
    else:
        raise 'Data directories for dataset ' + dataset + ' are not defined'
    return train_dir, val_dir
