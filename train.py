import argparse
import os
import time
import numpy as np
import copy as cp
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils.metrics import Evaluator, AverageMeter, accuracy
from utils.loss import SegmentationLosses
from dataset import make_data_loader
from networks import PAN, ResNet50
from utils.helpers import get_size_dataset, draw_curve, Kbar, split_fold, get_train_test_list

y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

x_epoch = []
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        opt (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, opt, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(opt, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


def parse_args():
    """Parse all the arguments provided from the CLI.

        Returns:
          A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Train a Pyramid Attention Networks for Land4Seen')

    parser.add_argument("--data_dir", type=str, default='./data/',
                        help="dataset path.")
    parser.add_argument("--train_list", type=str, default='./data/train.txt',
                        help="training list file.")
    parser.add_argument("--test_list", type=str, default='./data/train.txt',
                        help="test list file.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of images sent to the network in one step.")
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of iterations to train')
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help='starting learning rate')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="regularisation parameter for L2-loss")
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether use CUDA')
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread data-loading")
    parser.add_argument("--k_fold", type=int, default=10,
                        help="number of fold for k-fold.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="where to save snapshots of the modules.")

    return parser.parse_args()


def train(args, kbar, train_loader, convnet, pan, device, optimizer, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    running_loss = 0.0
    running_corrects = 0.0

    # modules.train() tells your modules that you are training the modules. This helps inform layers such as Dropout
    # and BatchNorm, which are designed to behave differently during training and evaluation. For instance,
    # in training mode, BatchNorm updates a moving average on each new batch;
    # whereas, for evaluation mode, these updates are frozen.
    convnet.train()
    pan.train()

    for batch_idx, batch in enumerate(train_loader):
        image, target, _, _ = batch
        inputs = image.to(device)
        labels = target.to(device)

        for md in [convnet, pan]:
            md.zero_grad()

        inputs = Variable(inputs)
        labels = Variable(labels)

        fms_blob, z = convnet(inputs)  # feature map, out
        out_ss = pan(fms_blob[::-1])
        out_ss = F.interpolate(out_ss, scale_factor=4, mode='nearest')

        loss_ss = criterion(out_ss, labels.long())
        acc_ss = accuracy(out_ss, labels.long())

        losses.update(loss_ss.item(), args.batch_size)
        scores.update(acc_ss.item(), args.batch_size)

        loss_ss.backward(torch.ones_like(loss_ss))
        for md in ['convnet', 'pan']:
            optimizer[md].step()

        # statistics
        running_loss += loss_ss.item()
        running_corrects += acc_ss.item()

        kbar.update(batch_idx, values=[("loss", loss_ss.item()), ("acc", 100. * scores.avg)])

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / len(train_loader)

    return OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
        ('epoch_loss', epoch_loss),
        ('epoch_acc', epoch_acc),
    ])


def validate(args, test_loader, convnet, pan, evaluator, device, criterion):
    convnet.eval()
    pan.eval()
    evaluator.reset()

    losses = AverageMeter()
    running_loss = 0.0
    running_corrects = 0.0

    for _, batch in enumerate(test_loader):
        image, target, _, _ = batch
        image = image.to(device)
        target = target.to(device)

        with torch.no_grad():
            fms_blob, z = convnet(image)
            out_ss = pan(fms_blob[::-1])

        out_ss = F.interpolate(out_ss, scale_factor=4, mode='nearest')
        loss_ss = criterion(out_ss, target.long())
        acc_ss = accuracy(out_ss, target.long())

        losses.update(loss_ss.item(), args.batch_size)

        # statistics
        running_loss += loss_ss.item()
        running_corrects += acc_ss.item()

        pred = out_ss.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = running_corrects / len(test_loader)

    # Fast test during the training
    acc = evaluator.Pixel_Accuracy()
    acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    return OrderedDict([
        ('loss', losses.avg),
        ('acc_pixel', acc),
        ('acc_class', acc_class),
        ('mIoU', mIoU),
        ('FWIoU', FWIoU),
        ('epoch_loss', epoch_loss),
        ('epoch_acc', epoch_acc)
    ])


def main():
    """Create the modules and start the training."""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Splitting k-fold
    split_fold(num_fold=args.k_fold, test_image_number=int(get_size_dataset('./data/img') / args.k_fold))

    # create modules
    convnet = ResNet50(pretrained=False)
    pan = PAN(blocks=convnet.blocks[::-1], num_class=args.num_classes)

    for fold in range(args.k_fold):
        print("\nTraining on fold %d" % fold)

        # Creating train.txt and test.txt
        get_train_test_list(fold)

        # create snapshots directory
        snapshot_dir = args.snapshot_dir + "fold" + str(fold)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        # Takes a local copy of the machine learning algorithm (modules) to avoid changing the one passed in
        convnet_ = cp.deepcopy(convnet)
        pan_ = cp.deepcopy(pan)

        convnet_.to(device)
        pan_.to(device)

        train_loader, test_loader = make_data_loader(args, num_workers=args.num_workers, pin_memory=True)

        optimizer = {'convnet': optim.Adam(convnet_.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                     'pan': optim.Adam(pan_.parameters(), lr=args.lr, weight_decay=args.weight_decay)}

        optimizer_lr_scheduler = {'convnet': PolyLR(optimizer['convnet'], max_iter=args.epochs, power=0.9),
                                  'pan': PolyLR(optimizer['pan'], max_iter=args.epochs, power=0.9)}

        criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')
        evaluator = Evaluator(num_class=args.num_classes)

        best_pred = 0.0
        train_per_epoch = round(get_size_dataset("./data/TrainData" + str(fold) + "/train/img/") / args.batch_size)

        for epoch in range(args.epochs):
            kbar = Kbar(target=train_per_epoch, epoch=epoch, num_epochs=args.epochs, width=25, always_stateful=False)

            train_log = train(args=args, kbar=kbar, train_loader=train_loader, convnet=convnet_, pan=pan_,
                              device=device, optimizer=optimizer, criterion=criterion)

            val_log = validate(args=args, test_loader=test_loader, convnet=convnet_, pan=pan_,
                               evaluator=evaluator, device=device, criterion=criterion)

            kbar.add(1, values=[("val_loss", val_log['loss']), ("val_acc", val_log['acc_pixel']),
                                ('acc_class', val_log['acc_class']), ('mIoU', val_log['mIoU']),
                                ('FWIoW', val_log['FWIoU'])])

            for md in ['convnet', 'pan']:
                optimizer_lr_scheduler[md].step()

            y_loss['train'].append(train_log['epoch_loss'])
            y_loss['val'].append(val_log['epoch_loss'])
            y_err['train'].append(1.0 - train_log['epoch_acc'])
            y_err['val'].append(1.0 - val_log['epoch_acc'])

            draw_curve(dir_save_fig=args.snapshot_dir, current_epoch=epoch + 1, x_epoch=x_epoch, y_loss=y_loss, y_err=y_err,
                       fig=fig, ax0=ax0, ax1=ax1)

            if val_log['mIoU'] > best_pred:
                print('\nEpoch %d: mIoU improved from %0.5f to %0.5f, saving model to %s' % (
                    epoch + 1, best_pred, val_log['mIoU'], args.snapshot_dir))
                best_pred = val_log['mIoU']
                torch.save(convnet.state_dict(), os.path.join(args.snapshot_dir, 'convnet.pth'))
                torch.save(pan.state_dict(), os.path.join(args.snapshot_dir, 'pan.pth'))
            else:
                print('\nEpoch %d: mIoU (%.05f) did not improve from %0.5f' % (epoch + 1, val_log['mIoU'], best_pred))


if __name__ == '__main__':
    main()
