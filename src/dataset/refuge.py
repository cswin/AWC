"""Dataset setting and data loader for USPS.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py.

use the test data as target data
added by PengLiu 12/08/2018
"""

import os
import numpy as np
import glob
import scipy.misc as m

import torch
from torch.autograd import Variable
import torch.utils.data as data

src_image_dir = '../data/processed/trainImage_save_path_600_aug'
src_mask_dir = '../data/processed/MaskImage_save_path_600_aug'
tgt_image_dir ='../data/processed/TestImage_save_path_460_aug/'
test_image_dir = '../data/processed/valiImage_save_path_460/'
test_mask_dir = '../data/processed/valiMaskImage_save_path_460/'


class REFUGE(data.Dataset):
    """REFUGE Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    def __init__(
        self, 
        train=True, 
        domain='REFUGE_SRC', 
        is_transform=False, 
        augmentations=None,
        aug_for_target=None,
        img_size=(400, 400),
        max_iters=None
    ):
        self.train = train
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.aug_for_target = aug_for_target
        self.dataset_size = None

        self.domain = domain
        if domain == 'REFUGE_SRC':
            self.img_dir = src_image_dir
            self.mask_dir = src_mask_dir
        if domain == 'REFUGE_DST':
            self.img_dir = tgt_image_dir
        if domain == 'REFUGE_TEST':
            self.img_dir = test_image_dir
            self.mask_dir = test_mask_dir

        self.class_map = {255: 0, 128: 1, 0: 2}
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )

        self._glob_img_files()
        if max_iters is not None:
            self.image_files = self.image_files * int(
                np.ceil(float(max_iters) / len(self.image_files)))

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_file = self.image_files[index]
        img = m.imread(image_file)
        img = np.array(img, dtype=np.uint8)

        if self.domain != 'REFUGE_DST':
            label_file = os.path.join(self.mask_dir,
                                      os.path.basename(image_file))[:-3] + 'bmp'
            lbl = m.imread(label_file)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        else:
            lbl = np.zeros(self.img_size, dtype=np.uint8)

        if self.augmentations is not None:
            if self.domain != 'REFUGE_DST':
                aug = self.augmentations(image=img, mask=lbl)
            else:
                aug = self.aug_for_target(image=img, mask=lbl)
            img0, lbl0 = aug['image'], aug['mask']
        else:
            img0, lbl0 = img.copy(), lbl.copy()

        if self.is_transform:
            img, lbl = self.transform(img,lbl)
            img0, lbl0 = self.transform(img0, lbl0)
        return img, lbl, img0, lbl0, os.path.basename(image_file)[:-4]

    def __len__(self):
        """Return size of dataset."""
        return len(self.image_files)

    def _glob_img_files(self):
        """Check if dataset is download and in right place."""
        if self.domain == 'REFUGE_SRC':
            self.image_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        if self.domain == 'REFUGE_DST':
            self.image_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        if self.domain == 'REFUGE_TEST':
            self.image_files = glob.glob(os.path.join(self.img_dir, '*.jpg'))

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)

        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]),
                         "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def encode_segmap(self, mask):
        # Put all void classes to zero
        classes = np.unique(mask)
        for each_class in classes:
            assert each_class in self.class_map.keys()

        for _validc in self.class_map.keys():
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def reapply_spatial_aug(self, img, lbl, spatial_augs):
        img_np = img.cpu().data.numpy().transpose(0, 2, 3, 1)
        img_np_aug = img_np.copy()
        lbl_np = lbl.data.numpy()
        lbl_aug = lbl_np.copy()
        n_img = img.shape[0]
        for idx in range(n_img):
            aug = spatial_augs[idx].call_again(image=img_np[idx],
                                               mask=lbl_np[idx])
            img_np_aug[idx], lbl_aug[idx] = aug['image'], aug['mask']
        img_np_aug = torch.from_numpy(img_np_aug.transpose(0, 3, 1, 2)).float()
        lbl_aug = torch.from_numpy(lbl_aug).long()
        img_np_aug = Variable(img_np_aug).cuda()
        lbl_aug = Variable(lbl_aug).cuda()
        return img_np_aug, lbl_aug
