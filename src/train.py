import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    OneOf,
    CLAHE,
    RandomGamma,
    HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise,
    RandomBrightnessContrast,
    IAASharpen, IAAEmboss
)

from models.unet import UNet
from models.discriminator import FCDiscriminator
from dataset.refuge import REFUGE
from pytorch_utils import (adjust_learning_rate, adjust_learning_rate_D,
                           calc_mse_loss, Weighted_Jaccard_loss)
from models import optim_weight_ema
from arguments import get_arguments


aug_student = Compose([
    OneOf([
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5)], p=0.2),

    OneOf([
        IAAAdditiveGaussianNoise(p=0.5),
        GaussNoise(p=0.5),
    ], p=0.2),

    OneOf([
        CLAHE(clip_limit=2),
        IAASharpen(p=0.5),
        IAAEmboss(p=0.5),
        RandomBrightnessContrast(p=0.5),
    ], p=0.2),
    HueSaturationValue(p=0.2),
    RandomGamma(p=0.2)])



aug_teacher = Compose([

    OneOf([
        IAAAdditiveGaussianNoise(p=0.5),
        GaussNoise(p=0.5),
    ], p=0.2),

    OneOf([
        CLAHE(clip_limit=2),
        IAASharpen(p=0.5),
        IAAEmboss(p=0.5),
        RandomBrightnessContrast(p=0.5),
    ], p=0.2),
    HueSaturationValue(p=0.2),
    RandomGamma(p=0.2)])


def main():
    """Create the model and start the training."""
    args = get_arguments()

    cudnn.enabled = True
    n_discriminators = 5

    # create teacher & student
    student_net = UNet(3, n_classes=args.num_classes)
    teacher_net = UNet(3, n_classes=args.num_classes)
    student_params = list(student_net.parameters())

    # teacher doesn't need gradient as it's just a EMA of the student
    teacher_params = list(teacher_net.parameters())
    for param in teacher_params:
        param.requires_grad = False

    student_net.train()
    student_net.cuda(args.gpu)
    teacher_net.train()
    teacher_net.cuda(args.gpu)

    cudnn.benchmark = True
    unsup_weights = [args.unsup_weight5, args.unsup_weight6, args.unsup_weight7,
                     args.unsup_weight8, args.unsup_weight9]
    lambda_adv_tgts = [args.lambda_adv_tgt5, args.lambda_adv_tgt6,
                       args.lambda_adv_tgt7, args.lambda_adv_tgt8,
                       args.lambda_adv_tgt9]

    # create a list of discriminators
    discriminators = []
    for dis_idx in range(n_discriminators):
        discriminators.append(FCDiscriminator(num_classes=args.num_classes))
        discriminators[dis_idx].train()
        discriminators[dis_idx].cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    max_iters = args.num_steps * args.iter_size * args.batch_size
    src_set = REFUGE(True, domain='REFUGE_SRC', is_transform=True,
                     augmentations=aug_student, aug_for_target=aug_teacher, max_iters=max_iters)
    src_loader = data.DataLoader(src_set,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    src_loader_iter = enumerate(src_loader)
    tgt_set = REFUGE(True, domain='REFUGE_DST', is_transform=True,
                     augmentations=aug_student, aug_for_target=aug_teacher,
                     max_iters=max_iters)
    tgt_loader = data.DataLoader(tgt_set,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    tgt_loader_iter = enumerate(tgt_loader)
    student_optimizer = optim.SGD(student_params,
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    teacher_optimizer = optim_weight_ema.WeightEMA(
        teacher_params, student_params, alpha=args.teacher_alpha)

    d_optimizers = []
    for idx in range(n_discriminators):
        optimizer = optim.Adam(discriminators[idx].parameters(),
                               lr=args.learning_rate_D,
                               betas=(0.9, 0.99))
        d_optimizers.append(optimizer)

    calc_bce_loss = torch.nn.BCEWithLogitsLoss()

    # labels for adversarial training
    source_label, tgt_label = 0, 1
    for i_iter in range(args.num_steps):

        total_seg_loss = 0
        seg_loss_vals = [0] * n_discriminators
        adv_tgt_loss_vals = [0] * n_discriminators
        d_loss_vals = [0] * n_discriminators
        unsup_loss_vals = [0] * n_discriminators

        for d_optimizer in d_optimizers:
            d_optimizer.zero_grad()
            adjust_learning_rate_D(d_optimizer, i_iter, args)

        student_optimizer.zero_grad()
        adjust_learning_rate(student_optimizer, i_iter, args)

        for sub_i in range(args.iter_size):

            # ******** Optimize source network with segmentation loss ********
            # As we don't change the discriminators, their parameters are fixed
            for discriminator in discriminators:
                for param in discriminator.parameters():
                    param.requires_grad = False

            _, src_batch = src_loader_iter.__next__()
            src_images, src_labels, _, _, _ = src_batch
            src_labels = Variable(src_labels).cuda(args.gpu)
            src_images = Variable(src_images).cuda(args.gpu)

            # calculate the segmentation losses
            sup_preds = list(student_net(src_images))
            seg_losses, total_seg_loss = [], 0
            for idx, sup_pred in enumerate(sup_preds):
                sup_interp_pred = (sup_pred)

                # [B, C_out, H_out, W_out] = sup_interp_pred.shape
                # sup_interp_pred = sup_interp_pred.view(B, H_out, W_out, C_out)

                seg_loss = Weighted_Jaccard_loss(src_labels, sup_interp_pred, args.gpu) #you also can use dice loss
                seg_losses.append(seg_loss)
                total_seg_loss += seg_loss * unsup_weights[idx]
                seg_loss_vals[idx] += seg_loss.item() / args.iter_size

            total_seg_loss = total_seg_loss / args.iter_size
            total_seg_loss.backward()
            student_optimizer.step()  # to have a trained model prepared for unsupevised learning and weigths adaptation.

            _, tgt_batch = tgt_loader_iter.__next__()
            tgt_images0, tgt_lbl0, tgt_images1, tgt_lbl1, _ = tgt_batch
            tgt_images0 = Variable(tgt_images0).cuda(args.gpu)
            tgt_images1 = Variable(tgt_images1).cuda(args.gpu)

            # As the requires_grad is set to False in the discriminator, the
            # gradients are only accumulated in the generator, the target
            # student network is optimized to make the outputs of target domain
            # images close to the outputs of source domain images
            stu_unsup_preds = list(student_net(tgt_images0))
            d_outs, loss = [], 0
            for idx in range(n_discriminators):
                stu_unsup_interp_pred = (stu_unsup_preds[idx])
                d_outs.append(discriminators[idx](stu_unsup_interp_pred))
                label_size = d_outs[idx].data.size()
                labels = torch.FloatTensor(label_size).fill_(source_label)
                labels = Variable(labels).cuda(args.gpu)
                adv_tgt_loss = calc_bce_loss(d_outs[idx], labels)

                loss += lambda_adv_tgts[idx] * adv_tgt_loss
                adv_tgt_loss_vals[idx] += adv_tgt_loss.item() / args.iter_size
            loss = loss / args.iter_size
            loss.backward()
            student_optimizer.step()

            # calculate ensemble losses
            stu_unsup_preds = list(student_net(tgt_images0))
            tea_unsup_preds = list(teacher_net(tgt_images1))
            loss_expr = 0
            for idx in range(n_discriminators):
                stu_unsup_probs = F.softmax(stu_unsup_preds[idx], dim=-1)
                tea_unsup_probs = F.softmax(tea_unsup_preds[idx], dim=-1)

                unsup_loss = calc_mse_loss(stu_unsup_probs, tea_unsup_probs, args.batch_size)
                unsup_loss_vals[idx] += unsup_loss.item() / args.iter_size
                loss_expr += unsup_loss * unsup_weights[idx]

            loss_expr = loss_expr / args.iter_size
            loss_expr.backward()
            student_optimizer.step()
            teacher_optimizer.step()

            # requires_grad is set to True in the discriminator,  we only
            # accumulate gradients in the discriminators, the discriminators are
            # optimized to make true predictions
            d_losses = []
            for idx in range(n_discriminators):
                discriminator = discriminators[idx]
                for param in discriminator.parameters():
                    param.requires_grad = True

                sup_preds[idx] = sup_preds[idx].detach()
                d_outs[idx] = discriminators[idx](sup_preds[idx])

                label_size = d_outs[idx].data.size()
                labels = torch.FloatTensor(label_size).fill_(source_label)
                labels = Variable(labels).cuda(args.gpu)

                d_losses.append(calc_bce_loss(d_outs[idx], labels))
                d_losses[idx] = d_losses[idx] / args.iter_size / 2
                d_losses[idx].backward()
                d_loss_vals[idx] += d_losses[idx].item()

            for idx in range(n_discriminators):
                stu_unsup_preds[idx] = stu_unsup_preds[idx].detach()
                d_outs[idx] = discriminators[idx](stu_unsup_preds[idx])

                label_size = d_outs[idx].data.size()
                labels = torch.FloatTensor(label_size).fill_(tgt_label)
                labels = Variable(labels).cuda(args.gpu)

                d_losses[idx] = calc_bce_loss(d_outs[idx], labels)
                d_losses[idx] = d_losses[idx] / args.iter_size / 2
                d_losses[idx].backward()
                d_loss_vals[idx] += d_losses[idx].item()

        for d_optimizer in d_optimizers:
            d_optimizer.step()

        log_str = 'iter = {0:7d}/{1:7d}'.format(i_iter, args.num_steps)
        log_str += ', total_seg_loss = {0:.3f} '.format(total_seg_loss)
        templ = 'seg_losses = [' + ', '.join(['%.2f'] * len(seg_loss_vals))
        log_str += templ % tuple(seg_loss_vals) + '] '
        templ = 'ens_losses = [' + ', '.join(['%.5f'] * len(unsup_loss_vals))
        log_str += templ % tuple(unsup_loss_vals) + '] '
        templ = 'adv_losses = [' + ', '.join(['%.2f'] * len(adv_tgt_loss_vals))
        log_str += templ % tuple(adv_tgt_loss_vals) + '] '
        templ = 'd_losses = [' + ', '.join(['%.2f'] * len(d_loss_vals))
        log_str += templ % tuple(d_loss_vals) + '] '

        print(log_str)
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            filename = 'UNet' + str(args.num_steps_stop) + 'v13.pth'
            torch.save(teacher_net.cpu().state_dict(),
                       os.path.join(args.snapshot_dir, filename))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            filename = 'UNet' + str(i_iter) + 'v13.pth'
            torch.save(teacher_net.cpu().state_dict(),
                       os.path.join(args.snapshot_dir, filename))
            teacher_net.cuda(args.gpu)


if __name__ == '__main__':
    main()
