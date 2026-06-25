import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from models.FHKMNet import FHKMNet
from data import get_dataloader
from utils.utils import load_model_params, set_gpu


def gen_maps(model, test_loader, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, name) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)

                sal_f = sal_f.cuda()
                _, _, _, res = model(x=image, ref_x=sal_f, y=None, training=False)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                cv2.imwrite(os.path.join(target_dir, name[0]+'.png'), res*255)
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FHKMNet')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=352, help='testing image size')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./dataset/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./snapshot/fhkmnet', help='the path to save model params and log')
    parser.add_argument('--pred_map_root', type=str, default='./pred_map/', help='the path to save pred maps')
    parser.add_argument('--pvt_weights', type=str, default='./pvt_weights/pvt_v2_b2.pth', help='the path to pretrained backbone weights')

    opt = parser.parse_args()
    print(opt)

    set_gpu(opt.gpu_id)

    ref_model = FHKMNet(opt).cuda()
    params_path = os.path.join(opt.save_root, opt.model_name, 'Net_epoch_45.pth')
    ref_model = load_model_params(ref_model, params_path)

    test_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.num_workers, mode='test')
    target_dir = os.path.join(opt.pred_map_root, opt.model_name)
    gen_maps(ref_model, test_loader, target_dir)
