import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
from time import time
from tqdm import tqdm
from datetime import datetime
from utils_lr import adjust_lr
from utils.utils import set_gpu, structure_loss, clip_gradient
from models.FHKMNet import FHKMNet
from data import get_dataloader
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.utils import load_model_params

import inspect


def save_model_code_to_txt(model, save_path):
    """
    Save source code of the instantiated model and its custom dependencies
    into a single txt file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    saved_files = set()

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('Model source code snapshot\n')
        f.write('=' * 80 + '\n\n')

        for m in model.modules():
            cls = m.__class__
            try:
                file_path = inspect.getfile(cls)
            except TypeError:
                continue

            if 'site-packages' in file_path:
                continue

            if file_path in saved_files:
                continue
            saved_files.add(file_path)

            try:
                src = inspect.getsource(cls)
            except OSError:
                continue

            f.write(f'# File: {file_path}\n')
            f.write('-' * 80 + '\n')
            f.write(src)
            f.write('\n\n')

    print(f'[INFO] Model code saved to {save_path}')

def compute_multiscale_structure_loss(preds, target):
    weights = [0.25, 0.25, 0.25, 0.25]
    return sum(w * structure_loss(pred, target) for w, pred in zip(weights, preds))

def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, supp_feats, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            supp_feats = supp_feats.cuda()
            s3, s2, s1, s0 = model(images, supp_feats, y=gts, training=True)
            loss = compute_multiscale_structure_loss((s3, s2, s1, s0), gts)
            loss_s3 = structure_loss(s3, gts)
            loss_s2 = structure_loss(s2, gts)
            loss_s1 = structure_loss(s1, gts)
            loss_s0 = structure_loss(s0, gts)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            writer.add_scalar('Loss-step', loss.item(), global_step=step)
            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_s3:{:.4f} loss_s2:{:.4f} '
                      'loss_s1:{:.4f} loss_s0:{:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_s3.data,
                             loss_s2.data, loss_s1.data, loss_s0.data))

        loss_all /= epoch_step
        writer.add_scalar('Loss/epoch', loss_all, global_step=epoch)

        if epoch % 200 == 0 and epoch > 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch
            }, save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch
        }, save_path + 'Net_Interrupt_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

def val(test_loader, model, save_root):
    global best_mae, best_epoch
    mae = MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, _) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)
                sal_f = sal_f.cuda()

                _, _, _, res = model(x=image, ref_x=sal_f, y=None, training=False)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                mae.step(pred=res*255, gt=gt*255)
                pbar.update()

            mae = mae.get_results()["mae"].round(5)

        results = {"MAE": mae}
        writer.add_scalar('Val/MAE', mae, epoch)

        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch
                }, save_root + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

        file_path = os.path.join(save_root, 'results.txt')
        file = open(file_path, "a")
        file.write(opt.model_name + str(results) + '\n')
        file.close()
        return results

def test_all_un(test_loader, model, save_root):

    Sm = Smeasure()
    Em = Emeasure()
    Fm = Fmeasure()
    wFm = WeightedFmeasure()
    mae = MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for (image, gt, sal_f, _) in test_loader:
                image = image.cuda()
                gt = gt.numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)
                sal_f = sal_f.cuda()

                _, _, _, res = model(x=image, ref_x=sal_f, y=None, training=False)
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


def set_seed(seed=42):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='FHKMNet')
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--imgsize', type=int, default=384, help='training image size')
    parser.add_argument('--shot', type=int, default=5, help='number of referring images')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_root', type=str, default='./dataset/R2C7K', help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='./snapshot/fhkmnet/', help='the path to save model params and log')
    parser.add_argument('--pvt_weights', type=str, default='./pvt_weights/pvt_v2_b2.pth', help='the path to save model params and log')
    set_seed(42)
    opt = parser.parse_args()
    print(opt)

    set_gpu(opt.gpu_id)
    cudnn.benchmark = True
    start_time = time()

    print(">>> before model init")
    model = FHKMNet(opt).cuda()
    print(">>> after model init")
    save_model_code_to_txt(
        model,
        os.path.join(opt.save_root, 'model_code.txt')
    )
    best_mae = 1
    best_epoch = 0
    base, body = [], []

    for name, param in model.named_parameters():
        if 'resnet' in name:
            base.append(param)
        else:
            body.append(param)

    params_dict = [{'params': base, 'lr': opt.lr * 0.1}, {'params': body, 'lr': opt.lr}]

    optimizer = torch.optim.Adam(params_dict)
    print('load data...')
    train_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, opt.batchsize, opt.num_workers, mode='train')
    test_loader = get_dataloader(opt.data_root, opt.shot, opt.imgsize, num_workers=opt.num_workers, mode='test')

    total_step = len(train_loader)
    save_path = opt.save_root + opt.model_name + '/'
    save_logs_path = opt.save_root + 'logs/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_logs_path, exist_ok=True)

    writer = SummaryWriter(save_logs_path + '/tb')
    step = 0

    print("Start train...")
    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(test_loader, model, save_path)

    ref_model = FHKMNet(opt).cuda()
    params_path = os.path.join(opt.save_root, '{}'.format(opt.model_name), 'Net_epoch_best.pth')
    ref_model = load_model_params(ref_model, params_path)

    scores = test_all_un(test_loader, ref_model, save_path)
    print(opt.save_root)
    end_time = time()
    print('it costs {} h to train'.format((end_time - start_time)/3600))
