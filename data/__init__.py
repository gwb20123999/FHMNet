"""
Data loading entrypoints for Ref-COD training and evaluation.
"""

import numpy as np
import torch
import torch.utils.data as data

from data.refdataset import RefCODDataset


def refcod_collate_fn(batch):
    images, gts, ref_feats, names = zip(*batch)
    images = torch.stack(images)
    ref_feats = torch.stack(ref_feats)

    processed_gts = []
    for gt in gts:
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt.astype(np.float32))
        elif isinstance(gt, torch.Tensor):
            gt = gt.to(torch.float32).cpu()
        else:
            raise TypeError(f"Unsupported gt type: {type(gt)}")
        processed_gts.append(gt)

    gts = torch.stack(processed_gts)
    return images, gts, ref_feats, list(names)


def get_dataloader(data_root, shot, trainsize, batchsize=32, num_workers=8, mode='train'):
    if mode == 'train':
        print('load train data...')
        dataset = RefCODDataset(
            data_root=data_root,
            mode='train',
            shot=shot,
            image_size=trainsize,
        )
        return data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=True,
            collate_fn=refcod_collate_fn,
        )

    if mode in ['val', 'test']:
        print('laod val data...')
        dataset = RefCODDataset(
            data_root=data_root,
            mode=mode,
            shot=shot,
            image_size=trainsize,
        )
        return data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            collate_fn=refcod_collate_fn,
        )

    raise KeyError(f'mode {mode} error!')
