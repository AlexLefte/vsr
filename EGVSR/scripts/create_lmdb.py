import os
import os.path as osp
import argparse
import glob
import lmdb
import pickle
import random
import numpy as np
import cv2


def split_spatial_temporal(frames, split_ratio):
    """Split a sequence into num_splits equal spatial-temporal parts."""
    _, h, w, _ = frames.shape
    splits = []
    rows, cols = map(int, split_ratio.split("_")) 
    for i in range(rows * cols):  # 4x4 = 16 secvențe
        sub_seq = frames[:, (i // cols) * h // rows: ((i // cols) + 1) * h // rows, 
                          (i % cols) * w // cols: ((i % cols) + 1) * w // cols, :]
        splits.append(sub_seq)
    return splits

def create_lmdb_with_splits(dataset, raw_dir, lmdb_dir, filter_file='', split_ratio='4_4'):
    # scan dir
    if filter_file:  # use sequences specified by the filter_file
        with open(filter_file, 'r') as f:
            seq_idx_lst = sorted([line.strip() for line in f])
    else:  # use all found sequences
        seq_idx_lst = sorted(os.listdir(raw_dir))

    num_seq = len(seq_idx_lst)
    print(f'>> Number of sequences: {num_seq}')

    # compute space to be allocated
    nbytes = 0
    for seq_idx in seq_idx_lst:
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED)
        # DELETE!!!
        h, w, _ = frm.shape
        frm = cv2.resize(frm, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
        nbytes_per_frm = frm.nbytes
        nbytes += len(frm_path_lst) * nbytes_per_frm
    alloc_size = round(2 * nbytes)
    print(f'>> Space required for lmdb generation: {alloc_size / (1 << 30):.2f} GB')

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 5
    keys = []
    txn = env.begin(write=True)
    sub_idx = 0
    for b, seq_idx in enumerate(seq_idx_lst):
        # log
        print(f'   Processing sequence: {seq_idx} ({b + 1}/{num_seq})\r', end='')

        # Get info
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        n_frm = len(frm_path_lst)

        # Read frames and split into spatial-temporal sequences
        frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in frm_path_lst]
        # DELETE!!!!
        h, w, _ = frames[0].shape
        frames = [cv2.resize(frm, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC) for frm in frames]
        frames = np.stack(frames)
        sub_sequences = split_spatial_temporal(frames, split_ratio)
        
        # Store each sub-sequence separately
        for sub_seq in sub_sequences:   
            n_frm = sub_seq.shape[0]
            for i in range(n_frm):      
                frm = sub_seq[i]
                frm = np.ascontiguousarray(frm[..., ::-1])
                h, w, _ = frm.shape
                key = f'{sub_idx}_{seq_idx}_{n_frm}x{h}x{w}_{i:04d}'  # Unique key with split index and shape
                txn.put(key.encode('ascii'), frm)
                keys.append(key)
            sub_idx += 1

            # commit
            if b % commit_freq == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.commit()
    env.close()

    # create meta information
    meta_info = {
        'name': dataset,
        'color': 'RGB',
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))

    print(f'>> Finished lmdb generation for {dataset}')


def create_lmdb(dataset, raw_dir, lmdb_dir, filter_file=''):
    assert dataset in ('VimeoTecoGAN', 'REDS'), f'Unknown Dataset: {dataset}'
    print(f'>> Start to create lmdb for {dataset}')

    # scan dir
    if filter_file:  # use sequences specified by the filter_file
        with open(filter_file, 'r') as f:
            seq_idx_lst = sorted([line.strip() for line in f])
    else:  # use all found sequences
        seq_idx_lst = sorted(os.listdir(raw_dir))

    num_seq = len(seq_idx_lst)
    print(f'>> Number of sequences: {num_seq}')

    # compute space to be allocated
    nbytes = 0
    for seq_idx in seq_idx_lst:
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        frm = cv2.imread(frm_path_lst[0], cv2.IMREAD_UNCHANGED)
        nbytes_per_frm = frm.nbytes
        nbytes += len(frm_path_lst) * nbytes_per_frm
    alloc_size = round(2 * nbytes)
    print(f'>> Space required for lmdb generation: {alloc_size / (1 << 30):.2f} GB')

    # create lmdb environment
    env = lmdb.open(lmdb_dir, map_size=alloc_size)

    # write data to lmdb
    commit_freq = 5
    keys = []
    txn = env.begin(write=True)
    for b, seq_idx in enumerate(seq_idx_lst):
        # log
        print(f'   Processing sequence: {seq_idx} ({b + 1}/{num_seq})\r', end='')

        # get info
        frm_path_lst = sorted(glob.glob(osp.join(raw_dir, seq_idx, '*.png')))
        n_frm = len(frm_path_lst)

        # read frames
        for i in range(n_frm):
            frm = cv2.imread(frm_path_lst[i], cv2.IMREAD_UNCHANGED)
            frm = np.ascontiguousarray(frm[..., ::-1])  # hwc|rgb|uint8

            h, w, c = frm.shape
            key = f'{seq_idx}_{n_frm}x{h}x{w}_{i:04d}'

            txn.put(key.encode('ascii'), frm)
            keys.append(key)

        # commit
        if b % commit_freq == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    # create meta information
    meta_info = {
        'name': dataset,
        'color': 'RGB',
        'keys': keys
    }
    pickle.dump(meta_info, open(osp.join(lmdb_dir, 'meta_info.pkl'), 'wb'))

    print(f'>> Finished lmdb generation for {dataset}')


def check_lmdb(dataset, lmdb_dir):

    def visualize(win, img):
        cv2.namedWindow(win, 0)
        cv2.resizeWindow(win, img.shape[-2], img.shape[-3])
        cv2.imshow(win, img[..., ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    assert dataset in ('VimeoTecoGAN', 'REDS'), f'Unknown Dataset: {dataset}'
    print(f'>> Start to check lmdb dataset: {dataset}.lmdb')

    # load keys
    meta_info = pickle.load(open(osp.join(lmdb_dir, 'meta_info.pkl'), 'rb'))
    keys = meta_info['keys']
    print(f'>> Number of keys: {len(keys)}')

    # randomly select frames for visualization
    with lmdb.open(lmdb_dir) as env:
        for i in range(3):  # can be replaced to any number
            idx = random.randint(0, len(keys) - 1)
            key = keys[idx]

            # parse key
            key_lst = key.split('_')
            vid, sz, frm = '_'.join(key_lst[:-2]), key_lst[-2], key_lst[-1]
            sz = tuple(map(int, sz.split('x')))
            sz = (*sz[1:], 3)
            print(f'   Visualizing frame: #{frm} from sequence: {vid} (size: {sz})')

            with env.begin() as txn:
                buf = txn.get(key.encode('ascii'))
                val = np.frombuffer(buf, dtype=np.uint8).reshape(*sz) # hwc

            visualize(key, val)

    print(f'>> Finished lmdb checking for {dataset}')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Vimeo90K")
    parser.add_argument('--raw_dir', type=str, default="data/Vimeo90K/raw/train",
                        help='Dir to the raw data')
    parser.add_argument('--lmdb_dir', type=str, default="data/Vimeo90K/lmdb/V90K_train_GT.lmdb",
                        help='Dir to the lmdb data')
    parser.add_argument('--filter_file', type=str, default='',
                        help='File used to select sequences')
    parser.add_argument('--split_ratio', type=str, default='1_1',
                        help='Provided as "h_w". Split ratio for saved clips: (h_orig // h) X (w_orig // w)')
    parser.add_argument('--degradation', type=str, default=None, help="Degradation type")
    parser.add_argument('--split', action='store_true', help="Enable split mode if this flag is present")
    args = parser.parse_args()

    # run
    if args.split:
        create_lmdb_with_splits(args.dataset, args.raw_dir, args.lmdb_dir, args.filter_file, args.split_ratio)
    else:
        create_lmdb(args.dataset, args.raw_dir, args.lmdb_dir, args.filter_file)
