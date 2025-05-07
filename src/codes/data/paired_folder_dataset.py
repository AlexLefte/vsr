import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files


import os
import os.path as osp
import random
import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files, retrieve_random_file


def augment_sequence(gt_pats, lr_pats):
    # flip
    axis = random.randint(1, 3)
    if axis > 1:
        gt_pats = np.flip(gt_pats, axis)
        lr_pats = np.flip(lr_pats, axis)

    # rotate 90 degree
    k = random.randint(0, 3)
    gt_pats = np.rot90(gt_pats, k, (2, 3))
    lr_pats = np.rot90(lr_pats, k, (2, 3))

    return gt_pats, lr_pats


class PairedFolderDatasetTrain(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(PairedFolderDatasetTrain, self).__init__(data_opt, **kwargs)

        # get keys
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        lr_keys = sorted(os.listdir(self.lr_seq_dir))
        self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = { line.strip() for line in f }
                self.keys = sorted(list(sel_keys & set(self.keys)))
            
        # GT bit depth
        self.gt_dtype = np.uint8 if self.gt_bit_depth == 8 else np.uint16

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load gt frames
        gt_seq, lr_seq = [], []

        # Augment data 
        if self.moving_first_frame and  \
            (random.uniform(0, 1) > self.moving_factor):
            # Read a random frame from the sequence
            gt_frm_path, lr_frm_path = retrieve_random_file(self.gt_seq_dir, self.lr_seq_dir, key)
            gt_frm = cv2.imread(gt_frm_path, cv2.IMREAD_UNCHANGED).transpose((2, 0, 1))
            lr_frm = cv2.imread(lr_frm_path, cv2.IMREAD_UNCHANGED).transpose((2, 0, 1))

            # Generate random moving parameters
            offsets = np.floor(
                np.random.uniform(-1.5, 1.5, size=(self.tempo_extent, 2)))
            offsets = offsets.astype(np.int32)
            pos = np.cumsum(offsets, axis=0)
            min_pos = np.min(pos, axis=0)
            topleft_pos = pos - min_pos
            range_pos = np.max(pos, axis=0) - min_pos
            _, lr_h, lr_w = lr_frm.shape
            c_h, c_w = lr_h - range_pos[0], lr_w - range_pos[1]
            s = self.scale

            # Generate frames based on the first frame
            for i in range(self.tempo_extent):
                lr_top, lr_left = topleft_pos[i]
                frm = lr_frm[:, lr_top: lr_top + c_h, lr_left: lr_left + c_w].copy()
                lr_seq.append(frm)

                gt_top, gt_left = lr_top * s, lr_left * s
                frm = gt_frm[:, gt_top: gt_top + c_h * s, gt_left: gt_left + c_w * s].copy()
                gt_seq.append(frm)
        else:
            # Pick the neccessary number of frames to form a sequence
            gt_files = retrieve_files(osp.join(self.gt_seq_dir, key))
            lr_files = retrieve_files(osp.join(self.lr_seq_dir, key))

            # Ensure we don't try to sample more frames than available
            num_frames = min(self.tempo_extent, len(gt_files))

            # Select a random starting index
            start_idx = random.randint(0, num_frames - self.tempo_extent)

            # Select the final sequences
            selected_gt_files = gt_files[start_idx: start_idx + self.tempo_extent]
            selected_lr_files = lr_files[start_idx: start_idx + self.tempo_extent]

            # Load selected GT frames
            for frm_path in selected_gt_files:
                frm = cv2.imread(frm_path, cv2.IMREAD_UNCHANGED)[..., ::-1]
                frm = np.transpose(frm, (2, 0, 1))
                gt_seq.append(frm)

            # Load selected LR frames
            for frm_path in selected_lr_files:
                frm = cv2.imread(frm_path)[..., ::-1]
                frm = np.transpose(frm, (2, 0, 1))
                lr_seq.append(frm)

        # Stack
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8
        lr_seq = np.stack(lr_seq)  # thwc|rgb|float32

        # Crop and augment sequences randomly if training
        gt_seq, lr_seq = self.crop_sequence(gt_seq, lr_seq)
        gt_seq, lr_seq = augment_sequence(gt_seq, lr_seq)

        # convert to tensor
        gt_tsr = torch.FloatTensor(np.ascontiguousarray(gt_seq)) / (2 ** self.gt_bit_depth - 1)
        lr_tsr = torch.FloatTensor(np.ascontiguousarray(lr_seq)) / 255.0

        # gt: thwc|rgb||uint | lr: thwc|rgb|float32
        return {
            'gt': gt_tsr,
            'lr': lr_tsr,
            'seq_idx': key,
            # 'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }

    def crop_sequence(self, gt_frms, lr_frms):
        gt_csz = self.gt_crop_size
        lr_csz = self.gt_crop_size // self.scale

        lr_h, lr_w = lr_frms.shape[-2:]
        assert (lr_csz <= lr_h) and (lr_csz <= lr_w), \
            'the crop size is larger than the image size'

        # crop lr
        lr_top = random.randint(0, lr_h - lr_csz)
        lr_left = random.randint(0, lr_w - lr_csz)
        lr_pats = lr_frms[
            ..., lr_top: lr_top + lr_csz, lr_left: lr_left + lr_csz]

        # crop gt
        gt_top = lr_top * self.scale
        gt_left = lr_left * self.scale
        gt_pats = gt_frms[
            ..., gt_top: gt_top + gt_csz, gt_left: gt_left + gt_csz]

        return gt_pats, lr_pats
    

class PairedFolderDatasetInfer(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(PairedFolderDatasetInfer, self).__init__(data_opt, **kwargs)

        # get keys
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        lr_keys = sorted(os.listdir(self.lr_seq_dir))
        self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = { line.strip() for line in f }
                self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        # load gt frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            frm = cv2.imread(frm_path, cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / \
                (2 ** self.gt_bit_depth - 1)
            gt_seq.append(frm.transpose(2, 0, 1))
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # load lr frames
        lr_seq = []
        for frm_path in retrieve_files(osp.join(self.lr_seq_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1].astype(np.float32) / 255.0
            lr_seq.append(frm.transpose(2, 0, 1))
        lr_seq = np.stack(lr_seq)  # thwc|rgb|float32

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq))  # float32

        # gt: thwc|rgb||uint8 | lr: thwc|rgb|float32
        return {
            'gt': gt_tsr,
            'lr': lr_tsr,
            'seq_idx': key,
            'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }
