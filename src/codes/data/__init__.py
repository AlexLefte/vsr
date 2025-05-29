import lmdb
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .paired_lmdb_dataset import PairedLMDBDataset
from .paired_folder_dataset import PairedFolderDatasetInfer, PairedFolderDatasetTrain


def is_lmdb(path):
    """Check if the given path is an LMDB database."""
    if not os.path.isdir(path):
        return False
    try:
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as txn:
            _ = txn.stat()  # Check if it has a valid LMDB structure
        return True
    except lmdb.Error:
        return False

def create_dataloader(opt, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)
    data_opt['gt_bit_depth'] = opt.get('gt_bit_depth', 8)
    degradation_type = opt['dataset']['degradation']['type']

    # -------------- loader for training -------------- #
    if dataset_idx == 'train':
        if degradation_type == 'BI':
            # create dataset
            # dataset = PairedLMDBDatasetV2(
            gt_path = data_opt['gt_seq_dir']
            if is_lmdb(gt_path):
                dataset = PairedLMDBDataset(data_opt,
                                            scale=opt['scale'],
                                            tempo_extent=opt['train']['tempo_extent'],
                                            moving_first_frame=opt['train'].get('moving_first_frame', False),
                                            moving_factor=opt['train'].get('moving_factor', 1.0),
                                            train=True)
            else:
                dataset = PairedFolderDatasetTrain(
                    data_opt,
                    scale=opt['scale'],
                    tempo_extent=opt['train']['tempo_extent'],
                    moving_first_frame=opt['train'].get('moving_first_frame', False),
                    moving_factor=opt['train'].get('moving_factor', 1.0),
                    train=True)

        elif degradation_type == 'BD':
            raise NotImplementedError('Unimplemented degradation type: {}'.format(
                degradation_type))
        
        else:
            raise ValueError('Unrecognized degradation type: {}'.format(
                degradation_type))

        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=data_opt['batch_size'],
            shuffle=True,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    # -------------- loader for testing -------------- #
    elif dataset_idx.startswith('test'):
        # create data loader
        gt_path = data_opt['gt_seq_dir']
        if is_lmdb(gt_path):
            # dataset = PairedLMDBDatasetV2(data_opt, 
            dataset = PairedLMDBDataset(data_opt, 
                                        scale=opt['scale'],
                                        tempo_extent=opt['test']['tempo_extent'],
                                        train=False)
        elif os.path.isdir(gt_path):
            dataset = PairedFolderDatasetInfer(data_opt, scale=opt['scale'])
        else:
            raise ValueError(f'Unrecognized GT type: {gt_path} is not a directory, nor an LMDB database.')
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return loader


def prepare_data(opt, data, kernel, has_lr=False):
    """ prepare gt, lr data for training

        for BD degradation, generate lr data and remove border of gt data
        for BI degradation, return data directly

    """

    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    if has_lr or degradation_type == 'BI':
        gt_data, lr_data = data['gt'].to(device), data['lr'].to(device)

    elif degradation_type == 'BD':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        border_size = int(sigma * 3.0)

        gt_with_border = data['gt'].to(device)
        n, t, c, gt_h, gt_w = gt_with_border.size()
        lr_h = (gt_h - 2 * border_size) // scale
        lr_w = (gt_w - 2 * border_size) // scale

        # generate lr data
        gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)
        lr_data = F.conv2d(
            gt_with_border, kernel, stride=scale, bias=None, padding=0)
        lr_data = lr_data.view(n, t, c, lr_h, lr_w)

        # remove gt border
        gt_data = gt_with_border[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt_data = gt_data.view(n, t, c, scale * lr_h, scale * lr_w)

    else:
        raise ValueError('Unrecognized degradation type: {}'.format(
            degradation_type))

    return { 'gt': gt_data, 'lr': lr_data }
