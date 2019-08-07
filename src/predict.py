import argparse
import numpy as np
from packaging import version
import os
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data

from models.unet import UNet
from dataset.refuge import REFUGE

NUM_CLASSES = 3
NUM_STEPS = 512  # Number of images in the validation set.
RESTORE_FROM = '../data/processed/snapshots/UNet_150000.pth'
SAVE_PATH = '../data/interim/GTA5_spacial_temporal_v13_04210623_150000'
BATCH_SIZE = 5

palette = [
    255, 255, 255,  # black background
    128, 128, 128,  # index 1 is red
    0,     0,   0,  # index 2 is yellow
    0,     0,   0   # index 3 is orange
]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unet Network")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size.")
    parser.add_argument("--gpu", type=int, default=3,
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = UNet(3, n_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.cuda(args.gpu)
    model.train()
    test_set = REFUGE(False, domain='REFUGE_TEST', is_transform=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(460, 460), mode='bilinear',
                             align_corners=True)
    else:
        interp = nn.Upsample(size=(460, 460), mode='bilinear')

    for index, batch in enumerate(test_loader):
        if index % 100 == 0:
            print('%d images are processed' % index)
        image, label, _, _, name = batch
        _, _, _, _, preds = model(Variable(image, volatile=True).cuda(args.gpu))
        output = interp(preds).cpu().data.numpy()

        for idx, one_name in enumerate(name):
            pred = output[idx]
            pred = pred.transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
            output_col = colorize_mask(pred)

            one_name = one_name.split('/')[-1]
            output_col = output_col.convert('L')
            output_col.save('%s/%s.bmp' % (args.save, one_name.split('.')[0]))


if __name__ == '__main__':
    main()
