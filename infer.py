import os
import argparse
import torch
import time

from PIL import Image
from torchvision.utils import save_image
from data.dataloader import ImageTransform
from models.sa_gan import STRnet2

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def init_args():
    print('init args...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOfWorkers', type=int, default=0,
                        help='workers for dataloader')
    parser.add_argument('--modelsSavePath', type=str, default='',
                        help='path for saving models')
    parser.add_argument('--logPath', type=str,
                        default='')
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--loadSize', type=int, default=512,
                        help='image loading size')
    parser.add_argument('--dataRoot', type=str,
                        default='./example')
    parser.add_argument('--pretrained',type=str, default='./model_path/model.pth', help='pretrained models for finetuning')
    parser.add_argument('--savePath', type=str, default='./results')
    args = parser.parse_args()
    return args

def init_model(args):
    print('init model...')
    savePath = args.savePath
    result_with_mask = savePath + '/WithMaskOutput/'
    result_straight = savePath + '/StrOuput/'
    # import pdb;pdb.set_trace()

    if not os.path.exists(savePath):
        os.makedirs(savePath)
        os.makedirs(result_with_mask)
        os.makedirs(result_straight)

    model = STRnet2(3)
    model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE))
    model.eval()
    return model

if __name__ == '__main__':
    args = init_args()
    print('args: {}'.format(args))
    model = init_model(args)
    print('model: {}'.format(model))
    img_path = './example/all_images/118.jpg'
    img = Image.open(img_path)
    img_trains = ImageTransform(args.loadSize)
    img = img_trains(img.convert('RGB'))
    with torch.no_grad():
        start = time.time()
        img = img.to(DEVICE)
        img = img.unsqueeze(0)
        out1, out2, out3, g_imgs, mm = model(img)
        g_imge = g_imgs.data.cpu()
        save_image(g_imge, args.savePath + '/result.jpg')


