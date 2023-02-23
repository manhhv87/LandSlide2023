import torch
import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for lp, lt in zip(pre_image, gt_image):
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        # >>> Initialize a meter to record loss
        # >>>     losses = AverageMeter()
        # >>> Update meter after every minibatch update
        # >>>     losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logit, target, top_k=(1,), ignore_idx=255):
    """ Suppose you have the ground truth prediction tensor y of shape b-h-w (dtype=torch.int64).
        Your modules predict per-pixel class logit of shape b-c-h-w, with c is the number of
        classes (including "background"). These logit are the "raw" predictions before softmax
        function transforms them into class probabilities. Since we are only looking at the top k,
        it does not matter if the predictions are "raw" or "probabilities".
    """
    max_k = max(top_k)

    # compute the top k predicted classes, per pixel
    _, tk = torch.topk(logit, max_k, dim=1)

    # you now have k predictions per pixel, and you want that one of them
    # will match the true labels target
    correct_pixels = torch.eq(target[:, None, ...], tk).any(dim=1).float()

    # take the mean of correct_pixels to get the overall average top-k accuracy
    valid = target != ignore_idx
    top_k_acc = correct_pixels[valid].mean()

    return top_k_acc
