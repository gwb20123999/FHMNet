import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.FHKMNet import FHKMNet
from data import get_dataloader
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.utils import load_model_params, set_gpu

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def save_pred_map(logit, out_hw, save_path, norm=True):
    """
    logit: torch.Tensor, shape [B,1,h,w] or [1,h,w] or [h,w]
    out_hw: (H,W) target size aligned with gt
    save_path: output .png path
    norm: whether to apply min-max normalization
    """
    if logit.dim() == 2:
        logit = logit.unsqueeze(0).unsqueeze(0)
    elif logit.dim() == 3:
        logit = logit.unsqueeze(0)
        if logit.shape[1] != 1:
            logit = logit.unsqueeze(1)
    elif logit.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected logit dim: {logit.dim()}")

    pred = torch.sigmoid(logit)
    pred = F.interpolate(pred, size=out_hw, mode='bilinear', align_corners=False)
    pred = pred[0, 0].detach().cpu().numpy()

    if norm:
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    pred_u8 = (pred * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, pred_u8)

def test_model(test_loader, model, save_root):

    Sm = Smeasure()
    Em = Emeasure()
    Fm = Fmeasure()
    wFm = WeightedFmeasure()
    mae = MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, name) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)

                sal_f = sal_f.cuda()
                s3, s2, s1, res = model(x=image, ref_x=sal_f, y=None, training=False)

                out_hw = gt.shape
                base_dir = os.path.join(save_root, "pred_maps")
                fname = os.path.splitext(os.path.basename(name[0] if isinstance(name, (list, tuple)) else name))[0]
                save_pred_map(res, out_hw, os.path.join(base_dir, "res", f"{fname}.png"), norm=True)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                Sm.step(pred=res*255, gt=gt*255)
                Em.step(pred=res*255, gt=gt*255)
                Fm.step(pred=res*255, gt=gt*255)
                wFm.step(pred=res*255, gt=gt*255)
                mae.step(pred=res*255, gt=gt*255)

                pbar.update()

            sm = Sm.get_results()["sm"]
            em = Em.get_results()["em"]
            fm = Fm.get_results()["fm"]
            wfm = wFm.get_results()["wfm"]
            mae = mae.get_results()["mae"]

        results = {
            "Smeasure": sm.round(5),
            "WeightedFmeasure": wfm.round(5),
            "MAE": mae.round(5),
            "adpEm": em["adp"].round(5),
            "meanEm": em["curve"].mean().round(5),
            "maxEm": em["curve"].max().round(5),
            "adpFm": fm["adp"].round(5),
            "meanFm": fm["curve"].mean().round(5),
            "maxFm": fm["curve"].max().round(5)
        }
        print(results)
        file_path = os.path.join(save_root, 'results.txt')
        file = open(file_path, "a")
        file.write(str(results) + '\n')
        file.close()
        return results
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FHKMNet')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=384, help='testing image size')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./dataset/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./snapshot/fhkmnet/', help='the path to save model params and log')
    parser.add_argument('--pvt_weights', type=str, default='./pvt_weights/pvt_v2_b2.pth', help='the path to pretrained backbone weights')
    opt = parser.parse_args()
    print(opt)
    set_gpu(opt.gpu_id)
    best_score = -1
    best_epoch = -1
    ref_model = FHKMNet(opt).cuda()

    params_path = os.path.join(opt.save_root, opt.model_name, 'Net_epoch_best.pth')
    ref_model = load_model_params(ref_model, params_path)
    test_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.num_workers, mode='test')
    scores = test_model(test_loader, ref_model, opt.save_root)
    print(scores)
