import os
import argparse
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from datasets import Voc2012
from networks import Classifier, PAN, ResNet50, Mask_Classifier, Color_Classifier
from utils import save_model, get_each_cls_iu, PolyLR
from sklearn.metrics import average_precision_score
import ss_transforms as tr

parser = argparse.ArgumentParser(description='PAN')
parser.add_argument('--checkpoints', type=str, default=None,
                    help='The directory of training model')
args = parser.parse_args()

experiment_name = "Test_{:}".format(args.checkpoints[:-4])
path_log = Path('./log/' + experiment_name + '.log')
try:
    if path_log.exists():
        raise FileExistsError
except FileExistsError:
    print("Already exist log file: {}".format(path_log))
    raise
else:
    logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S',
                                filename=path_log.__str__(),
                                filemode='w'
                                )
    print('Create log file: {}'.format(path_log))

test_transforms = transforms.Compose([tr.FixedResize((512, 512)),
                                      tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                      tr.ToTensor()
])

test_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012'
                    ,'val',transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

length_test_dataset = len(test_data)

NUM_CLASS = 20

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

convnet = ResNet50(pretrained=True)
classifier = Classifier(in_features=2048, num_class=NUM_CLASS)
pan = PAN(convnet.blocks[::-1])
mask_classifier = Mask_Classifier(in_features=256, num_class=(NUM_CLASS+1))
color_classifier = Color_Classifier(in_features=256, num_class=2)

convnet.to(device)
classifier.to(device)
pan.to(device)
mask_classifier.to(device)
color_classifier.to(device)

def test(data_loader):
    global best_acc
    convnet.eval()
    pan.eval()
    all_i_count = []
    all_u_count = []
    y_true = []
    y_pred = []
    loss_ic = 0
    pixel_acc = 0
    for batch_idx, (imgs, cls_labels, mask_labels) in enumerate(data_loader):
        with torch.no_grad():
            imgs, cls_labels = imgs.to(device), cls_labels.to(device)
            imgs_ab = imgs[:, 1:, :, :]
            fms_blob, out_convnet = convnet(imgs)
            out_cls = classifier(out_convnet.detach())
            out_pan = pan(fms_blob[::-1])
            mask_pred = mask_classifier(out_pan)
            mask_pred = F.interpolate(mask_pred, scale_factor=4, mode='nearest')

            # Image Colorization Loss
            color_pred = color_classifier(out_pan)
            color_pred = F.interpolate(color_pred, scale_factor=4, mode='nearest')
            loss_ic += F.mse_loss(color_pred, imgs_ab).item()

        # results
        y_pred.append(torch.sigmoid(out_cls).data.cpu().numpy())
        y_true.append(cls_labels.data.cpu().numpy())

        i_count, u_count = get_each_cls_iu(mask_pred.max(1)[1].cpu().data.numpy(), mask_labels.long().squeeze(1).numpy())
        all_i_count.append(i_count)
        all_u_count.append(u_count)
        pixel_acc += mask_pred.max(dim=1)[1].data.cpu().eq(mask_labels.long().squeeze(1)).float().mean().item()

    # Result
    acc = average_precision_score(np.concatenate(y_true, 0), np.concatenate(y_pred, 0))
    each_cls_IOU = (np.array(all_i_count).sum(0) / np.array(all_u_count).sum(0))
    mIOU = each_cls_IOU.mean()
    pixel_acc = pixel_acc / length_test_dataset
    loss_ic = loss_ic / length_test_dataset
    logging.info("Length of test set:{:} Test Cls Acc:{:.4f}% Each_cls_IOU:{:} mIOU:{:.4f} PA:{:.2f}% loss_ic{:.6f}".format(length_test_dataset, acc*100, dict(zip(test_data.classes, (100*each_cls_IOU).tolist())), mIOU*100, pixel_acc*100, loss_ic))


test(test_loader)