import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from collections import OrderedDict

from utils.metrics import *
from utils.loss import SegmentationLosses
from dataset import make_data_loader
from networks import PAN, ResNet50

y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}


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

    return parser.parse_args()


def train(args, train_loader, convnet, pan, device, optimizer, criterion):
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

    for _, batch in enumerate(train_loader):
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
    test_loss = 0.0

    losses = AverageMeter()
    scores = AverageMeter()
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

        # loss = loss_ss.item()
        # test_loss += loss
        losses.update(loss_ss.item(), args.batch_size)
        scores.update(acc_ss.item(), args.batch_size)

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

    # new_pred = mIoU
    # if new_pred > best_pred:
    #     best_pred = new_pred
    #     torch.save(convnet, './convnet.pth')
    #     torch.save(pan, './pan.pth')

    return OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
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
    print('device:{}'.format(device))

    train_loader, test_loader = make_data_loader(args, num_workers=args.num_workers, pin_memory=True)

    convnet = ResNet50(pretrained=False)
    pan = PAN(blocks=convnet.blocks[::-1], num_class=args.num_classes)

    convnet.to(device)
    pan.to(device)

    optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                 'pan': optim.Adam(pan.parameters(), lr=args.lr, weight_decay=args.weight_decay)}

    optimizer_lr_scheduler = {'convnet': PolyLR(optimizer['convnet'], max_iter=args.epochs, power=0.9),
                              'pan': PolyLR(optimizer['pan'], max_iter=args.epochs, power=0.9)}

    criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')
    evaluator = Evaluator(num_class=args.num_classes)

    best_pred = 0.0
    for epoch in range(args.epochs):
        tem_time = time.time()

        for m in ['convnet', 'pan']:
            optimizer_lr_scheduler[m].step(epoch)

        train_log = train(args=args, train_loader=train_loader, convnet=convnet, pan=pan,
                          device=device, optimizer=optimizer, criterion=criterion)

        val_log = validate(args=args, test_loader=test_loader, convnet=convnet, pan=pan,
                           evaluator=evaluator, device=device, criterion=criterion)

        # Gather data and report
        epoch_time = time.time() - tem_time

        y_loss['train'].append(train_log['epoch_loss'])
        y_loss['val'].append(val_log['epoch_loss'])
        y_err['train'].append(1.0 - train_log['epoch_acc'])
        y_err['val'].append(1.0 - val_log['epoch_acc'])

        if val_log['mIoU'] > best_pred:
            best_pred = val_log['mIoU']
            torch.save(convnet, './convnet.pth')
            torch.save(pan, './pan.pth')

        # Reports the loss for each epoch
        print(
            'Epoch %d/%d - %.2fs - loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f '
            '- acc_pixel %.4f - acc_class %.4f - mIoU %.4f - FWIoU %.4f' %
            (epoch + 1, args.epochs, epoch_time, train_log['loss'], train_log['acc'], val_log['loss'],
             val_log['acc'], val_log['acc_pixel'], val_log['acc_class'], val_log['mIoU'], val_log['FWIoU']))


if __name__ == '__main__':
    main()
