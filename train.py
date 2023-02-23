import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from networks import PAN, ResNet50

from dataset import make_data_loader
import argparse
from utils.metrics import *
from utils.loss import SegmentationLosses
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from collections import OrderedDict


class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

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

    parser.add_argument('--epochs', dest='epochs',
                        help='number of iterations to train',
                        default=50, type=int)

    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=True, type=bool)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument('--gpu_ids', dest='gpu_ids',
                        help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)',
                        default='0', type=str)

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='batch_size')

    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, uint is epoch',
                        default=50, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    return parser.parse_args()


NUM_CLASS = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:{}'.format(device))
args = parse_args()
kwargs = {'num_workers': 0, 'pin_memory': True}
train_loader, test_loader = make_data_loader(args, **kwargs)

convnet = ResNet50(pretrained=False)
pan = PAN(blocks=convnet.blocks[::-1], num_class=NUM_CLASS)

convnet.to(device)
pan.to(device)

criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')

model_name = ['convnet', 'pan']
optimizer = {'convnet': optim.Adam(convnet.parameters(), lr=args.lr, weight_decay=1e-4),
             'pan': optim.Adam(pan.parameters(), lr=args.lr, weight_decay=1e-4)}

optimizer_lr_scheduler = {'convnet': PolyLR(optimizer['convnet'], max_iter=args.epochs, power=0.9),
                          'pan': PolyLR(optimizer['pan'], max_iter=args.epochs, power=0.9)}

evaluator = Evaluator(NUM_CLASS)


def train(epoch, optimizer, train_loader):
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

    for iteration, batch in enumerate(train_loader):
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

        # if iteration % 10 == 0:
        #     print("Epoch[{}]({}/{}):Loss:{:.4f}".format(epoch, iteration, len(train_loader),
        #                                                 loss_ss.data))

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / len(train_loader)

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
        ('epoch_loss', epoch_loss),
        ('epoch_acc', epoch_acc),
    ])

    return log


def validation(epoch, best_pred):
    convnet.eval()
    pan.eval()
    evaluator.reset()
    test_loss = 0.0
    for iteration, batch in enumerate(test_loader):
        image, target, _, _ = batch
        image = image.to(device)
        target = target.to(device)
        with torch.no_grad():
            fms_blob, z = convnet(image)
            out_ss = pan(fms_blob[::-1])
        out_ss = F.interpolate(out_ss, scale_factor=4, mode='nearest')
        loss_ss = criterion(out_ss, target.long())
        loss = loss_ss.item()
        test_loss += loss
        print('epoch:{},test loss:{}'.format(epoch, test_loss / (iteration + 1)))

        pred = out_ss.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * args.batch_size + image.shape[0]))
    print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % test_loss)

    new_pred = mIoU
    if new_pred > best_pred:
        print('(mIoU)new pred ={},old best pred={}'.format(new_pred, best_pred))
        best_pred = new_pred
        torch.save(convnet, './convnet.pth')
        torch.save(pan, './pan.pth')
    return best_pred


y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

best_pred = 0.0
for epoch in range(args.epochs):
    for m in model_name:
        optimizer_lr_scheduler[m].step(epoch)
    print('Epoch:{}'.format(epoch))
    train_log = train(epoch, optimizer, train_loader)

    y_loss['train'].append(train_log['epoch_loss'])
    y_err['train'].append(1.0 - train_log['epoch_acc'])

    # Reports the loss for each epoch
    print('Epoch %d/%d - loss %.4f - acc %.4f' %
          (epoch + 1, args.epochs, train_log['loss'], train_log['acc']))

    if epoch % (5 - 1) == 0:
        best_pred = validation(epoch, best_pred)
