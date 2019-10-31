import argparse

import numpy as np

from packaging import version

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate
import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.utils import data

# from models.unet import UNet
from models.networks import R2AttU_Net
from dataset.refuge import REFUGE

NUM_CLASSES = 3
NUM_STEPS = 512 # Number of images in the validation set.
RESTORE_FROM = '../data/snapshots/UNet80000v21_JointOpt_multiScale_R2AttU_Net.pth'
SAVE_PATH = '../data/result_UNet80000v21_JointOpt_multiScale_R2AttU_Net/'
MODEL = 'Unet'
BATCH_SIZE = 1
is_polar = False  #If need to transfer the image and labels to polar coordinates: MICCAI version is False
ROI_size = 460    #ROI size

print(RESTORE_FROM)

palette=[
    255, 255, 255, # black background
    128, 128, 128, # index 1 is red
    0, 0, 0, # index 2 is yellow
    0, 0 , 0 # index 3 is orange
]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Unet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice Unet.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--is_polar", type=bool, default=False,
                        help="If proceed images in polar coordinate. MICCAI version is false")
    parser.add_argument("--ROI_size", type=int, default=460,
                        help="Size of ROI.")

    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    return parser.parse_args()



def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # model = UNet(3, n_classes=args.num_classes)
    model = R2AttU_Net(img_ch=3, output_ch=args.num_classes, t=args.t)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.cuda(gpu0)
    model.train()

    testloader = data.DataLoader(REFUGE(False, domain='REFUGE_TEST', is_transform=True),
                                    batch_size=args.batch_size, shuffle=False, pin_memory=True)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(460, 460), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(460, 460), mode='bilinear')

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % index)
        image, label, _, _, name = batch
        if args.model == 'Unet':
            _,_,_,_, output2  = model(Variable(image, volatile=True).cuda(gpu0))

            output = interp(output2).cpu().data.numpy()


        for idx, one_name in enumerate(name):
            pred = output[idx]
            pred = pred.transpose(1,2,0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
            output_col = colorize_mask(pred)

            if is_polar:

                # plt.imshow(output_col)
                # plt.show()


                output_col = np.array(output_col)
                output_col[output_col == 0] = 0
                output_col[output_col == 1] = 128
                output_col[output_col == 2] = 255

                # plt.imshow(output_col)
                # plt.show()

                output_col = cv2.linearPolar(rotate(output_col, 90), (args.ROI_size / 2, args.ROI_size / 2),
                                             args.ROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

                # plt.imshow(output_col)
                # plt.show()

                output_col = np.array(output_col * 255, dtype=np.uint8)
                output_col[output_col > 200] = 210
                output_col[output_col == 0] = 255
                output_col[output_col == 210] = 0
                output_col[(output_col > 0) & (output_col < 255)] = 128

                output_col = Image.fromarray(output_col)

                # plt.imshow(output_col)
                # plt.show()

            one_name = one_name.split('/')[-1]
            if len(one_name.split('_'))>1:
                one_name = one_name[:-4]
            #pred.save('%s/%s.bmp' % (args.save, one_name))
            output_col = output_col.convert('L')

            print(output_col.size)
            output_col.save('%s/%s.bmp' % (args.save, one_name.split('.')[0]))


if __name__ == '__main__':
    main()
