import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from .data.dataloader import ErasingData
from .loss.Loss import LossWithGAN_STE
from .models.Model import VGG16FeatureExtractor
from .models.sa_gan import STRnet2
import utils

torch.set_num_threads(5)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"    ### set the gpu as No....

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=500, help='epochs')
args = parser.parse_args()


def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

dataRoot = args.dataRoot

# import pdb;pdb.set_trace()
Erase_data = ErasingData(dataRoot, loadSize, training=True)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, 
                         shuffle=True, num_workers=args.numOfWorkers, drop_last=False, pin_memory=True)
print('=============', len(Erase_data))

# val dataset
val_data_root='/home/shiyan/data/image_hw_remo/dehw_val_dataset'
Erase_val_data = ErasingData(val_data_root, loadSize, training=False)
Erase_val_data = DataLoader(Erase_val_data, batch_size=1,
                         shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
print('==============', len(Erase_val_data))

netG = STRnet2(3)
if args.pretrained != '':
    print('loaded ')
    netG.load_state_dict(torch.load(args.pretrained, map_location=DEVICE))

netG = netG.to(DEVICE)

G_optimizer = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))

criterion = LossWithGAN_STE(args.logPath, VGG16FeatureExtractor(), lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)

criterion = criterion.to(DEVICE)

print('OK!')
num_epochs = args.num_epochs
best_psner = 0
count = 1
with torch.autograd.set_detect_anomaly(True):
    best_psnr = 0
    for i in range(1, num_epochs + 1):
        netG.train()
        for k,(imgs, gt, masks, path) in enumerate(Erase_data):
            imgs = imgs.to(DEVICE)
            gt = gt.to(DEVICE)
            masks = masks.to(DEVICE)

            netG.zero_grad()

            x_o1,x_o2,x_o3,fake_images,mm = netG(imgs)
            G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, count, i)
            G_loss = G_loss.sum()
            G_optimizer.zero_grad()

            G_loss = G_loss.detach_().requires_grad_(True)
            G_loss.backward()

            G_optimizer.step()

            print('[{}/{}] Generator Loss of epoch{} is {}'.format(k,len(Erase_data),i, G_loss.item()))

            count += 1

        netG.eval()
        val_psnr = 0
        for index, (imgs, gt, masks, path) in enumerate(Erase_val_data):
            print(index, imgs.shape, gt.shape, path)
            _, _, h, w = imgs.shape
            rh, rw = h, w
            step = 512
            pad_h = step - h if h < step else 0
            pad_w = step - w if w < step else 0
            m = nn.ZeroPad2d((0, pad_w, 0, pad_h))
            imgs = m(imgs)
            _, _, h, w = imgs.shape
            res = torch.zeros_like(imgs)
            for i in range(0, h, step):
                for j in range(0, w, step):
                    if h - i < step:
                        i = h - step
                    if w - j < step:
                        j = w - step
                    clip = imgs[:, :, i:i + step, j:j + step]
                    clip = clip.to(DEVICE)
                    with torch.no_grad():
                        _, _, _, g_images_clip, mm = netG(clip)
                    g_images_clip = g_images_clip.cpu()
                    mm = mm.cpu()
                    clip = clip.cpu()
                    mm = torch.where(F.sigmoid(mm) > 0.5, torch.zeros_like(mm), torch.ones_like(mm))
                    g_image_clip_with_mask = clip * (mm) + g_images_clip * (1 - mm)
                    res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
            res = res[:, :, :rh, :rw]
            output = utils.pd_tensor2img(res)
            target = utils.pd_tensor2img(gt)
            del res
            del gt
            psnr = utils.compute_psnr(target, output)
            del target
            del output
            val_psnr += psnr
            print('index:{} psnr: {}'.format(index, psnr))
        ave_psnr = val_psnr / (index + 1)
        # torch.save(netG.state_dict(), args.modelsSavePath + '/STE_{}_{:.4f}.pth'.format(i, ave_psnr))
        if ave_psnr > best_psnr:
            best_psnr = ave_psnr
            torch.save(netG.state_dict(), args.modelsSavePath + '/STE_best.pth')
        print('epoch: {}, ave_psnr: {}, best_psnr: {}'.format(i, ave_psnr, best_psnr))

